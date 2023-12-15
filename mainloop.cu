/*
* TODO: finish refactoring random code -- search for getRandomRollback
* TODO: refactor statcounter to remove it as global variable
* TODO: think about how to manage stack counter when every thread is updating their own statcounter 
* -- maybe use a global statcounter and update it at the end of each thread's execution within main
* TODO: think about how to manage the clock counter when every thread is updating their own statcounter
*/

/* Frequency of comprehensive reports-- lower values will provide more
 * info while slowing down the simulation. Higher values will give less
 * frequent updates. */
/* This is also the frequency of screen refreshes if SDL is enabled. */
#define REPORT_FREQUENCY 200000

/* Mutation rate -- range is from 0 (none) to 0xffffffff (all mutations!) */
/* To get it from a float probability from 0.0 to 1.0, multiply it by
 * 4294967295 (0xffffffff) and round. */
#define MUTATION_RATE 5000

/* How frequently should random cells / energy be introduced?
 * Making this too high makes things very chaotic. Making it too low
 * might not introduce enough energy. */
#define INFLOW_FREQUENCY 100

/* Base amount of energy to introduce per INFLOW_FREQUENCY ticks */
#define INFLOW_RATE_BASE 600

/* A random amount of energy between 0 and this is added to
 * INFLOW_RATE_BASE when energy is introduced. Comment this out for
 * no variation in inflow rate. */
#define INFLOW_RATE_VARIATION 1000

/* Size of pond in X and Y dimensions. */
#define POND_SIZE_X 800
#define POND_SIZE_Y 600

/* Depth of pond in four-bit codons -- this is the maximum
 * genome size. This *must* be a multiple of 16! */
#define POND_DEPTH 1024

/* This is the divisor that determines how much energy is taken
 * from cells when they try to KILL a viable cell neighbor and
 * fail. Higher numbers mean lower penalties. */
#define FAILED_KILL_PENALTY 3

/*
* Buffer size is the size for the circular buffer
*/
#define BUFFER_SIZE 1000  // Size of the circular buffer

/* Pond depth in machine-size words.  This is calculated from
 * POND_DEPTH and the size of the machine word. (The multiplication
 * by two is due to the fact that there are two four-bit values in
 * each eight-bit byte.) */
#define POND_DEPTH_SYSWORDS (POND_DEPTH / (sizeof(uintptr_t) * 2))

/* Number of bits in a machine-size word */
#define SYSWORD_BITS (sizeof(uintptr_t) * 8)

/* Constants representing neighbors in the 2D grid. */
#define N_LEFT 0
#define N_RIGHT 1
#define N_UP 2
#define N_DOWN 3

/* Word and bit at which to start execution */
/* This is after the "logo" */
#define EXEC_START_WORD 0
#define EXEC_START_BIT 4
/* ----------------------------------------------------------------------- */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>


/**
 * Structure for a cell in the pond
 */
struct __align__(8) Cell
    {
	/* Globally unique cell ID */
	uint64_t ID;
	
	/* ID of the cell's parent */
	uint64_t parentID;
	
	/* Counter for original lineages -- equal to the cell ID of
	 * the first cell in the line. */
	uint64_t lineage;
	
	/* Generations start at 0 and are incremented from there. */
	uintptr_t generation;
	
	/* Energy level of this cell */
	uintptr_t energy;

	/* Memory space for cell genome (genome is stored as four
	 * bit instructions packed into machine size words) */
	uintptr_t genome[POND_DEPTH_SYSWORDS];
};

struct statCounters {
	/* Counts for the number of times each instruction was
	 * executed since the last report. */
	double instructionExecutions[16];
	
	/* Number of cells executed since last report */
	double cellExecutions;
	
	/* Number of viable cells replaced by other cells' offspring */
	uintptr_t viableCellsReplaced;
	
	/* Number of viable cells KILLed */
	uintptr_t viableCellsKilled;
	
	/* Number of successful SHARE operations */
	uintptr_t viableCellShares;
};

__host__ __device__ static inline void getRandomPre(int rollback, uintptr_t *ret, uint64_t *prngState)
{
    // https://en.wikipedia.org/wiki/Xorshift#xorshift.2B
    uint64_t x = prngState[0];
    const uint64_t y = prngState[1];
    prngState[0] = prngState[0] * !rollback + rollback * y;
    x ^= x << 23;
    const uint64_t z = x ^ y ^ (x >> 17) ^ (y >> 26);
    prngState[1] = prngState[1] * !rollback + rollback * z;
    *ret = (uintptr_t)(z + y);
}

void precalculate_random_numbers(uintptr_t *buffer, uint64_t *prngState) {
    for (int i = 0; i < BUFFER_SIZE; i++) {
        uintptr_t val;
        getRandomPre(1, &val, prngState);
        buffer[i] = val;
    }
}

__device__ static inline void getRandomRollback(uintptr_t rollback, uintptr_t *ret, uintptr_t *buffer, int *in, uint64_t *prngState) {
    uintptr_t num = buffer[*in];
    uintptr_t new_num;
    getRandomPre(rollback, &new_num, prngState);
    buffer[*in] = (new_num & -rollback) | (num & ~-rollback);
    *in = (((*in + 1) & -rollback) | (*in & ~-rollback)) % BUFFER_SIZE;
    *ret = num;
}

__device__ static inline void accessAllowed(struct Cell *const c2, const uintptr_t c1guess, int sense, int rollback, int *ret, uintptr_t *buffer, int *in, uint64_t *prngState)
{
    uintptr_t BITS_IN_FOURBIT_WORD[16] = { 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4 };
    uintptr_t random = 0; 
    getRandomRollback(rollback, &random, buffer, in, prngState);
    random = (uintptr_t)(random & 0xf);
    *ret = sense ? (((random & 0xf) >= BITS_IN_FOURBIT_WORD[(c2->genome[0] & 0xf) ^ (c1guess & 0xf)])||(!c2->parentID)) : (((random & 0xf) <= BITS_IN_FOURBIT_WORD[(c2->genome[0] & 0xf) ^ (c1guess & 0xf)])||(!c2->parentID));
    /* Access permission is more probable if they are more similar in sense 0,
     * and more probable if they are different in sense 1. Sense 0 is used for
     * "negative" interactions and sense 1 for "positive" ones. */
    //*ret = ((((random >= BITS_IN_FOURBIT_WORD[(c2->genome[0] & 0xf) ^ (c1guess & 0xf)]) || !c2->parentID) & sense) | (((random <= BITS_IN_FOURBIT_WORD[(c2->genome[0] & 0xf) ^ (c1guess & 0xf)]) || !c2->parentID) & ~sense));
}

__device__ static inline void getNeighbor(struct Cell *pond, const uintptr_t x, const uintptr_t y, const uintptr_t dir, struct Cell **ret)
{
    /* Define the changes in the x and y coordinates for each direction */
    int dx[] = {-1, 1, 0, 0}; // Changes in x for N_LEFT, N_RIGHT, N_UP, N_DOWN
    int dy[] = {0, 0, -1, 1}; // Changes in y for N_LEFT, N_RIGHT, N_UP, N_DOWN

    /* Calculate the new coordinates */
    uintptr_t newX = (x + dx[dir] + POND_SIZE_X) % POND_SIZE_X;
    uintptr_t newY = (y + dy[dir] + POND_SIZE_Y) % POND_SIZE_Y;

    *ret = &pond[newY * POND_SIZE_X + newX];
}

static void doReport(struct Cell *pond, struct statCounters *statCounter, const uint64_t clock)
{
    static uint64_t lastTotalViableReplicators = 0;
    
    uintptr_t x,y;
    
    uint64_t totalActiveCells = 0;
    uint64_t totalEnergy = 0;
    uint64_t totalViableReplicators = 0;
    uintptr_t maxGeneration = 0;
    
    for(x=0;x<POND_SIZE_X;++x) {
        for(y=0;y<POND_SIZE_Y;++y) {
            struct Cell *const c = &pond[y * POND_SIZE_X + x];
            if (c->energy) {
                ++totalActiveCells;
                totalEnergy += (uint64_t)c->energy;
                if (c->generation > 2)
                    ++totalViableReplicators;
                if (c->generation > maxGeneration)
                    maxGeneration = c->generation;
            }
        }
    }
	
	/* Look here to get the columns in the CSV output */
	
	/* The first five are here and are self-explanatory */
	printf("%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu",
		(uint64_t)clock,
		(uint64_t)totalEnergy,
		(uint64_t)totalActiveCells,
		(uint64_t)totalViableReplicators,
		(uint64_t)maxGeneration,
		(uint64_t)statCounter->viableCellsReplaced,
		(uint64_t)statCounter->viableCellsKilled,
		(uint64_t)statCounter->viableCellShares
		);
	
	/* The next 16 are the average frequencies of execution for each
	 * instruction per cell execution. */
	double totalMetabolism = 0.0;
	for(x=0;x<16;++x) {
		totalMetabolism += statCounter->instructionExecutions[x];
		printf(",%.4f",(statCounter->cellExecutions > 0.0) ? (statCounter->instructionExecutions[x] / statCounter->cellExecutions) : 0.0);
	}
	
	/* The last column is the average metabolism per cell execution */
	printf(",%.4f\n",(statCounter->cellExecutions > 0.0) ? (totalMetabolism / statCounter->cellExecutions) : 0.0);
	fflush(stdout);
	
	if ((lastTotalViableReplicators > 0)&&(totalViableReplicators == 0))
		fprintf(stderr,"[EVENT] Viable replicators have gone extinct. Please reserve a moment of silence.\n");
	else if ((lastTotalViableReplicators == 0)&&(totalViableReplicators > 0))
		fprintf(stderr,"[EVENT] Viable replicators have appeared!\n");
	
	lastTotalViableReplicators = totalViableReplicators;
	
	/* Reset per-report stat counters */
	for(x=0;x<sizeof(statCounter);++x)
		((uint8_t *)&statCounter)[x] = (uint8_t)0;
}

__global__ static void run(struct Cell *pond, uintptr_t *buffer, int *in, uint64_t *prngState, struct statCounters *statCounter, uint64_t *cellIdCounter, int *accessNeg, int *accessPos) 
{
    //const uintptr_t threadNo = (uintptr_t)targ;
    uintptr_t x,y,i;
    uintptr_t clock = 0;
    uintptr_t outputBuf[POND_DEPTH_SYSWORDS];
    uintptr_t currentWord,wordPtr,shiftPtr,inst,tmp;
    struct Cell *pptr, *tmpptr;
    uintptr_t ptr_wordPtr;
    uintptr_t ptr_shiftPtr;
    uintptr_t reg;
    uintptr_t facing;
    uintptr_t loopStack_wordPtr[POND_DEPTH];
    uintptr_t loopStack_shiftPtr[POND_DEPTH];
    uintptr_t loopStackPtr;
    uintptr_t falseLoopDepth;
    int stop;
    int skip;
    int access_neg_used;
    int access_pos_used;
    //uintptr_t access_neg;
    //uintptr_t access_pos;
    uintptr_t rand;
    int exitNow = 0;
    while (!exitNow) {
        clock++;
        //printf("clock: %lu\n", clock);
        if (clock == 10000)
        {
            exitNow = 1;
        }
        if (!(clock % INFLOW_FREQUENCY)) {
            getRandomRollback(1, &x, buffer, in, prngState);
            x = x % POND_SIZE_X;
            getRandomRollback(1, &y, buffer, in, prngState);
            y = y % POND_SIZE_Y;
            pptr = &pond[y * POND_SIZE_X + x];
            pptr->ID = *cellIdCounter;
            pptr->parentID = 0;
            pptr->lineage = *cellIdCounter;
            pptr->generation = 0;
    #ifdef INFLOW_RATE_VARIATION
            getRandomRollback(1, &rand, buffer, in, prngState);
            pptr->energy += INFLOW_RATE_BASE + (rand % INFLOW_RATE_VARIATION);
    #else
            pptr->energy += INFLOW_RATE_BASE;
    #endif /* INFLOW_RATE_VARIATION */
            for(i=0;i<POND_DEPTH_SYSWORDS;++i) 
                getRandomRollback(1, &rand, buffer, in, prngState);
                pptr->genome[i] = rand;
            ++(*cellIdCounter);
        }
        /* Pick a random cell to execute */
        getRandomRollback(1, &rand, buffer, in, prngState);
        //
        x = rand % POND_SIZE_X;
        y = ((rand / POND_SIZE_X) >> 1) % POND_SIZE_Y;
        pptr = &pond[y * POND_SIZE_X + x];
        /* Reset the state of the VM prior to execution */
        for(i=0;i<POND_DEPTH_SYSWORDS;++i)
            outputBuf[i] = ~((uintptr_t)0); /* ~0 == 0xfffff... */
        ptr_wordPtr = 0;
        ptr_shiftPtr = 0;
        reg = 0;
        loopStackPtr = 0;
        wordPtr = EXEC_START_WORD;
        shiftPtr = EXEC_START_BIT;
        facing = 0;
        falseLoopDepth = 0;
        stop = 0;
        skip=0;
        access_neg_used = 0;
        access_pos_used = 0;
        //access_neg = 0;
        //access_pos = 0;
        /* We use a currentWord buffer to hold the word we're
            * currently working on.  This speeds things up a bit
            * since it eliminates a pointer dereference in the
            * inner loop. We have to be careful to refresh this
            * whenever it might have changed... take a look at
            * the code. :) */
        currentWord = pptr->genome[0];
        /* Keep track of how many cells have been executed */
        statCounter->cellExecutions += 1.0;
        /* Core execution loop */
        while ((pptr->energy)&&(!stop)) {
            /* Get the next instruction */
            inst = (currentWord >> shiftPtr) & 0xf;
            skip=0;
            /* Randomly frob either the instruction or the register with a
                * probability defined by MUTATION_RATE. This introduces variation,
                * and since the variation is introduced into the state of the VM
                * it can have all manner of different effects on the end result of
                * replication: insertions, deletions, duplications of entire
                * ranges of the genome, etc. */
            getRandomRollback(1, &rand, buffer, in, prngState);
            if ((rand & 0xffffffff) < MUTATION_RATE) {
                getRandomRollback(1, &rand, buffer, in, prngState);
                tmp = rand; // Call getRandom() only once for speed 
                if (tmp & 0x80) // Check for the 8th bit to get random boolean 
                    inst = tmp & 0xf; // Only the first four bits are used here 
                else reg = tmp & 0xf;
            }
            /* Each instruction processed costs one unit of energy */
            --pptr->energy;
            /* Execute the instruction */
            if (falseLoopDepth) {
                /* Skip forward to matching REP if we're in a false loop. */
                if (inst == 0x9) /* Increment false LOOP depth */
                    ++falseLoopDepth;
                else if (inst == 0xa) /* Decrement on REP */
                    --falseLoopDepth;
            } else {
                ptr_shiftPtr = (inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xd || inst == 0xe || inst == 0xf) * (ptr_shiftPtr) +((inst == 0x0)*0)+((inst == 0x1)*((ptr_shiftPtr+4)*((ptr_shiftPtr+4)<SYSWORD_BITS)))+((inst == 0x2)*(((ptr_shiftPtr==0)*SYSWORD_BITS)+ptr_shiftPtr-4)); 
                ptr_wordPtr = (inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xd || inst == 0xe || inst == 0xf) * (ptr_wordPtr) +((inst == 0x0)*0)+((inst == 0x1)*(((ptr_wordPtr*(ptr_shiftPtr!=0||((ptr_wordPtr+1)<POND_DEPTH_SYSWORDS))+(ptr_shiftPtr==0)*((ptr_wordPtr+1)<POND_DEPTH_SYSWORDS)))))+((inst == 0x2)*(((ptr_wordPtr==0&&ptr_shiftPtr==(SYSWORD_BITS-4))*(POND_DEPTH_SYSWORDS))+ptr_wordPtr-(ptr_shiftPtr==(SYSWORD_BITS-4))));
                wordPtr=(inst==0x0||inst==0x1||inst==0x2||inst==0x3||inst==0x4||inst==0x5|| inst == 0x6 || inst==0x7||inst==0x8||inst==0x9||inst==0xb||inst==0xd||inst==0xe||inst==0xf)*(wordPtr)+((inst==0xa)*(wordPtr*!(reg&&loopStackPtr)+(loopStack_wordPtr[loopStackPtr-1]*(reg&&loopStackPtr))))+((inst==0xc)*(wordPtr*((shiftPtr+4<SYSWORD_BITS)||(wordPtr+1<POND_DEPTH_SYSWORDS))+((shiftPtr+4>=SYSWORD_BITS)&&(wordPtr+1<POND_DEPTH_SYSWORDS))+EXEC_START_WORD*((wordPtr+1>=POND_DEPTH_SYSWORDS)&&(shiftPtr+4>=SYSWORD_BITS)))); 
                shiftPtr=(inst==0x0||inst==0x1||inst==0x2||inst==0x3||inst==0x4||inst==0x5|| inst == 0x6 || inst==0x7||inst==0x8||inst==0x9||inst==0xb||inst==0xd||inst==0xe||inst==0xf)*(shiftPtr)+((inst==0xa)*(shiftPtr*!(reg&&loopStackPtr)+(loopStack_shiftPtr[loopStackPtr-1]*(reg&&loopStackPtr))))+((inst==0xc)*((shiftPtr+4)+(shiftPtr+4>=SYSWORD_BITS)*(-shiftPtr-4)));
                skip=(reg&&loopStackPtr)*(inst==0xa);
                facing=(inst==0x1||inst==0x2||inst==0x3||inst==0x4||inst==0x5||inst==0x6||inst==0x7||inst==0x8||inst==0x9||inst==0xa||inst==0xc||inst==0xd||inst==0xe||inst==0xf)*(facing) + ((inst==0x0)*0)+((inst==0xb)*(reg & 3));
                pptr->genome[ptr_wordPtr]=(inst==0x0||inst==0x1||inst==0x2||inst==0x3||inst==0x4||inst==0x5||inst==0x7||inst==0x8||inst==0x9||inst==0xa||inst==0xb||inst==0xc||inst==0xd||inst==0xe||inst==0xf)*(pptr->genome[ptr_wordPtr])+((inst==0x6)*((pptr->genome[ptr_wordPtr]&~(((uintptr_t)0xf)<<ptr_shiftPtr))|reg<<ptr_shiftPtr)); 
                outputBuf[ptr_wordPtr]=(inst==0x0||inst==0x1||inst==0x2||inst==0x3||inst==0x4||inst==0x5|| inst == 0x6 || inst==0x7||inst==0x9||inst==0xa||inst==0xb||inst==0xc||inst==0xd||inst==0xe||inst==0xf)*(outputBuf[ptr_wordPtr])+((inst==0x8)*((outputBuf[ptr_wordPtr]&~(((uintptr_t)0xf) << ptr_shiftPtr))|reg << ptr_shiftPtr));
                currentWord=(inst==0x0||inst==0x1||inst==0x2||inst==0x3||inst==0x4||inst==0x5||inst==0x7||inst==0x8||inst==0x9|| inst==0xb || inst==0xd||inst==0xe||inst==0xf)*(currentWord)+((inst==0x6)*(pptr->genome[wordPtr]))+((inst==0xa)*(currentWord*!(reg&&loopStackPtr)+(pptr->genome[wordPtr])*(reg&&loopStackPtr)))+((inst == 0xc)*(pptr->genome[wordPtr]));
                stop=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xd || inst == 0xe)*(stop)+((inst == 0x9)*(stop*!(reg&&(loopStackPtr>=POND_DEPTH))+(reg&&(loopStackPtr>=POND_DEPTH))))+((inst == 0xf)*(1));
                loopStack_wordPtr[loopStackPtr]=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xd || inst == 0xe || inst == 0xf)*(loopStack_wordPtr[loopStackPtr])+((inst == 0x9)*(loopStack_wordPtr[loopStackPtr]*(!reg||(loopStackPtr>=POND_DEPTH))+(wordPtr*(reg&&(loopStackPtr<POND_DEPTH)))));
                loopStack_shiftPtr[loopStackPtr]=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xd || inst == 0xe || inst == 0xf)*(loopStack_shiftPtr[loopStackPtr])+((inst == 0x9) * (loopStack_shiftPtr[loopStackPtr]*(!reg||(loopStackPtr>=POND_DEPTH))+(shiftPtr*(reg&&(loopStackPtr<POND_DEPTH)))));
                loopStackPtr=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0xb || inst == 0xc || inst == 0xd || inst == 0xe || inst == 0xf)*(loopStackPtr)+((inst == 0x9)*(loopStackPtr + (reg&&(loopStackPtr<POND_DEPTH))))+((inst == 0xa)*(loopStackPtr-!!loopStackPtr));
                falseLoopDepth=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xd || inst == 0xe || inst == 0xf)*(falseLoopDepth)+((inst == 0x9)*(falseLoopDepth + (!reg)));
                getNeighbor(pond, x,y,facing, &tmpptr);
                access_neg_used = 0;
                access_pos_used = 0;
                access_pos_used = (inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xd || inst == 0xf)*(access_pos_used)+((inst == 0xe)*(1));
                access_neg_used = (inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xe|| inst == 0xf)*(access_neg_used)+((inst == 0xd)*(1));
                accessAllowed(tmpptr,reg,0, access_neg_used, accessNeg, buffer, in, prngState);
                accessAllowed(tmpptr,reg,1, access_pos_used, accessPos, buffer, in, prngState);
                statCounter->viableCellsKilled=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xf)*(statCounter->viableCellsKilled)+((inst == 0xd)*(statCounter->viableCellsKilled+(*accessNeg)*(tmpptr->generation>2)))+((inst == 0xe)*(statCounter->viableCellsKilled+(*accessPos)*(tmpptr->generation>2)));
                tmpptr->genome[0]=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xe || inst == 0xf)*(tmpptr->genome[0])+((inst == 0xd)*(tmpptr->genome[0]*!(*accessNeg)+(*accessNeg)*~((uintptr_t)0)));
                tmpptr->genome[1]=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xe || inst == 0xf)*(tmpptr->genome[1])+((inst == 0xd)*(tmpptr->genome[0]*!(*accessNeg)+(*accessNeg)*~((uintptr_t)0)));
                tmpptr->ID=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xe || inst == 0xf)*(tmpptr->ID)+((inst == 0xd)*(tmpptr->ID * !(*accessNeg)+ (*accessNeg)*(*cellIdCounter)));
                tmpptr->parentID=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xe || inst == 0xf)*(tmpptr->parentID)+((inst == 0xd)*(tmpptr->parentID * !(*accessNeg)));
                tmpptr->lineage=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xe || inst == 0xf)*(tmpptr->lineage)+((inst == 0xd)*(tmpptr->lineage * !(*accessNeg) + (*accessNeg)*(*cellIdCounter)));
                *cellIdCounter=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xe || inst == 0xf)*(*cellIdCounter)+((inst == 0xd)*(*cellIdCounter * !(*accessNeg) + (*accessNeg)* (*cellIdCounter)));
                tmp = (inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xf)*(tmp)+((inst == 0xc)*(reg))+((inst == 0xd)*((*accessNeg) + (tmpptr->generation>2)*!(*accessNeg)*(pptr->energy / FAILED_KILL_PENALTY)))+((inst == 0xe)* (pptr->energy + tmpptr->energy));
                reg = (inst == 0x1 || inst == 0x2 || inst == 0x6 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xd ||inst == 0xe || inst == 0xf) * (reg) + ((inst==0x0)*0) + ((inst==0x3)*((reg + 1) & 0xf)) +((inst==0x4)*((reg - 1) & 0xf)) +((inst==0x5)*((pptr->genome[ptr_wordPtr] >> ptr_shiftPtr) & 0xf)) +((inst==0x7)*((outputBuf[ptr_wordPtr] >> ptr_shiftPtr) & 0xf)) +((inst==0xc)*((pptr->genome[wordPtr] >> shiftPtr) & 0xf));
                pptr->genome[wordPtr]= (inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xd || inst == 0xe || inst == 0xf)*(pptr->genome[wordPtr])+((inst == 0xc)* (((pptr->genome[wordPtr]&~(((uintptr_t)0xf) << shiftPtr))|tmp << shiftPtr)));
                pptr->energy = (inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xf)*(pptr->energy)+((inst == 0xd)*(pptr->energy+!(*accessNeg)*(tmpptr->generation>2)*(-pptr->energy) + !(*accessNeg)*(tmpptr->generation>2)*(pptr->energy-tmp)))+((inst == 0xe)*((*accessPos * (tmp - (*accessPos * (tmp / 2) + (1 - *accessPos) * tmpptr->energy)) + (1 - *accessPos) * pptr->energy)));
                tmpptr->generation = (inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xe || inst == 0xf)*(tmpptr->generation)+((inst == 0xd)*(tmpptr->generation * (*accessNeg)));
                tmpptr->energy=(inst == 0x0 || inst == 0x1 || inst == 0x2 || inst == 0x3 || inst == 0x4 || inst == 0x5 || inst == 0x6 || inst == 0x7 || inst == 0x8 || inst == 0x9 || inst == 0xa || inst == 0xb || inst == 0xc || inst == 0xd || inst == 0xf)*(tmpptr->energy)+((inst == 0xe)*((*accessPos * (tmp / 2) + (1 - *accessPos) * tmpptr->energy)));
                /* Keep track of execution frequencies for each instruction */
                statCounter->instructionExecutions[inst] += 1.0;
            }
            wordPtr=(wordPtr*((shiftPtr+4<SYSWORD_BITS)||(wordPtr+1<POND_DEPTH_SYSWORDS)) + ((shiftPtr+4>=SYSWORD_BITS)&&(wordPtr+1<POND_DEPTH_SYSWORDS)) + EXEC_START_WORD*((wordPtr+1>=POND_DEPTH_SYSWORDS)&&(shiftPtr+4>=SYSWORD_BITS)))*!skip + wordPtr*skip;
            currentWord=(currentWord*(shiftPtr+4<SYSWORD_BITS)+(pptr->genome[wordPtr])*(shiftPtr+4>=SYSWORD_BITS))*!skip+ currentWord*skip;
            shiftPtr=((shiftPtr+4)+(shiftPtr+4>=SYSWORD_BITS)*(-shiftPtr-4))*!skip+shiftPtr*skip;
        }   
        /* Copy outputBuf into neighbor if access is permitted and there
            * is energy there to make something happen. There is no need
            * to copy to a cell with no energy, since anything copied there
            * would never be executed and then would be replaced with random
            * junk eventually. See the seeding code in the main loop above. */
        if ((outputBuf[0] & 0xff) != 0xff) {
            getNeighbor(pond,x,y,facing, &tmpptr);
            //printf("%lu\n", tmpptr->energy);
            if ((tmpptr->energy)) {
                //int t;
                accessAllowed(tmpptr,reg,0,1, accessPos, buffer, in, prngState);
                if(accessPos) {
                    /* Log it if we're replacing a viable cell */
                    if (tmpptr->generation > 2)
                        ++statCounter->viableCellsReplaced;
                    tmpptr->ID = ++(*cellIdCounter);
                    tmpptr->parentID = pptr->ID;
                    tmpptr->lineage = pptr->lineage; /* Lineage is copied in offspring */
                    tmpptr->generation = pptr->generation + 1;
                    
                    for(i=0;i<POND_DEPTH_SYSWORDS;++i)
                        tmpptr->genome[i] = outputBuf[i];
                }
            }
        }
    }
}

__global__ void initializePond(struct Cell *pond) {
    int x = blockIdx.x;
    int y = threadIdx.x;

    pond[x * POND_SIZE_Y + y].ID = 0;
    pond[x * POND_SIZE_Y + y].parentID = 0;
    pond[x * POND_SIZE_Y + y].lineage = 0;
    pond[x * POND_SIZE_Y + y].generation = 0;
    pond[x * POND_SIZE_Y + y].energy = 0;
    for(int i = 0; i < POND_DEPTH_SYSWORDS; ++i)
        pond[x * POND_SIZE_Y + y].genome[i] = ~((uintptr_t)0);
}

int main() {
    // Declare device pointers
    uintptr_t *d_buffer;
    int *d_in;
    uintptr_t *d_last_random_number;
    uint64_t *d_prngState;
    int *d_accessAllowed1;
    int *d_accessAllowed2;
    /* This is used to generate unique cell IDs */
    uint64_t *cellIdCounter;

    cudaMalloc(&d_accessAllowed1, sizeof(int));
    cudaMalloc(&d_accessAllowed2, sizeof(int));
    // Allocate memory on the GPU for cellIdCounter
    cudaMalloc(&cellIdCounter, sizeof(uint64_t));

    // Set its value to 0
    cudaMemset(cellIdCounter, 0, sizeof(uint64_t));

    // Allocate memory on the GPU for each variable
    cudaMalloc(&d_buffer, BUFFER_SIZE * sizeof(uintptr_t));
    cudaMalloc(&d_in, sizeof(int));
    cudaMalloc(&d_last_random_number, sizeof(uintptr_t));
    cudaMalloc(&d_prngState, 2 * sizeof(uint64_t));

    // Allocate memory for buffer and prngState on the host
    uintptr_t *h_buffer = (uintptr_t *)malloc(BUFFER_SIZE * sizeof(uintptr_t));
    uint64_t h_prngState[2] = {0, (uint64_t)rand()};

    // Call precalculate_random_numbers
    precalculate_random_numbers(h_buffer, h_prngState);

    // Copy the updated buffer and prngState to the device
    cudaMemcpy(d_buffer, h_buffer, BUFFER_SIZE * sizeof(uintptr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prngState, h_prngState, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Allocate the pond
    struct Cell *d_pond;
    cudaMalloc(&d_pond, POND_SIZE_X * POND_SIZE_Y * sizeof(struct Cell));
    // ON CPU
    struct statCounters *statCounters = (struct statCounters *)malloc(sizeof(struct statCounters));
    struct Cell *h_pond = (struct Cell *)malloc(POND_SIZE_X * POND_SIZE_Y * sizeof(struct Cell));
    // Reset per-report stat counters
    // Declare a device pointer for statCounters
    struct statCounters *d_statCounters;
 
    cudaMalloc(&d_statCounters, sizeof(d_statCounters));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Clear the pond and initialize all genomes
    // This can be done in a kernel
    initializePond<<<POND_SIZE_X, POND_SIZE_Y>>>(d_pond);

   // Call the kernel function
    for (uint64_t n = 0; n < 10; n++){
        run<<<1, 1>>>(d_pond, d_buffer, d_in, d_prngState, d_statCounters, cellIdCounter, d_accessAllowed1, d_accessAllowed2);
        cudaDeviceSynchronize(); 
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        cudaMemcpy(statCounters, d_statCounters, sizeof(struct statCounters), cudaMemcpyDeviceToHost);  
        cudaMemcpy(h_pond, d_pond, POND_SIZE_X * POND_SIZE_Y * sizeof(struct Cell), cudaMemcpyDeviceToHost);
        doReport(h_pond, statCounters, n);
    }
    

    // Free the memory on the GPU
    cudaFree(d_buffer);
    cudaFree(d_in);
    cudaFree(d_last_random_number);
    cudaFree(d_prngState);
    cudaFree(d_pond);
    cudaFree(d_statCounters);
    free(statCounters);
    free(h_pond);
    cudaFree(d_accessAllowed1);
    cudaFree(d_accessAllowed2);
    return 0;
}
