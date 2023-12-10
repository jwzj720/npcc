#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0')


/* Depth of pond in four-bit codons -- this is the maximum
 * genome size. This *must* be a multiple of 16! */
#define POND_DEPTH 1024


/* ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>


/* Pond depth in machine-size words.  This is calculated from
 * POND_DEPTH and the size of the machine word. (The multiplication
 * by two is due to the fact that there are two four-bit values in
 * each eight-bit byte.) */
#define POND_DEPTH_SYSWORDS (POND_DEPTH / (sizeof(uintptr_t) * 2))

/* Number of bits in a machine-size word */
#define SYSWORD_BITS (sizeof(uintptr_t) * 8)

/* Word and bit at which to start execution */
/* This is after the "logo" */
#define EXEC_START_WORD 0
#define EXEC_START_BIT 4

#define POND_SIZE_X 6
#define POND_SIZE_Y 6

/* Number of bits set in binary numbers 0000 through 1111 */
static const uintptr_t BITS_IN_FOURBIT_WORD[16] = { 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4 };

/**
 * Structure for a cell in the pond
 */
struct Cell
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

static struct Cell pond[POND_SIZE_X][POND_SIZE_Y];

void bin(unsigned n)
{
    unsigned i;
    for (i = 1 << 31; i > 0; i = i / 2)
        (n & i) ? printf("1") : printf("0");
}

/* The pond is a 2D array of cells */
static struct Cell pond[POND_SIZE_X][POND_SIZE_Y];
#define GENOME_SIZE 1025

void printUnpackedCell(struct Cell cell) {
    uintptr_t wordPtr = 0;
    uintptr_t shiftPtr = 0;
    uintptr_t packedValue;

    while (wordPtr < sizeof(cell.genome) / sizeof(cell.genome[0])) {
        packedValue = cell.genome[wordPtr];
        while (shiftPtr < SYSWORD_BITS) {
            char character = (packedValue >> shiftPtr) & 0xF;
            printf("%c", character + '0');
            shiftPtr += 4;
        }
        wordPtr++;
        shiftPtr = 0;
    }
    printf("\n");
}
static struct Cell readCell(char *genomeData) {
    uintptr_t wordPtr = 0;
    uintptr_t shiftPtr = 0;
    uintptr_t packedValue = 0;
    struct Cell cell;

    for (int i = 0; genomeData[i] != '\0'; i++) {
        char character = genomeData[i];
        if (character >= '0' && character <= 'f') {
            uintptr_t value = character <= '9' ? character - '0' : character - 'a' + 10;
            packedValue |= value << shiftPtr;
            shiftPtr += 4;

            if (shiftPtr >= SYSWORD_BITS) {
                cell.genome[wordPtr] = packedValue;
                wordPtr++;
                shiftPtr = 0;
                packedValue = 0;
            }
        }
    }

    if (shiftPtr > 0) {
        cell.genome[wordPtr] = packedValue;
    }
    return cell;
}

static void writeCell(FILE *file, struct Cell *cell) {
    uintptr_t wordPtr,shiftPtr,inst,stopCount,i;
		wordPtr = 0;
		shiftPtr = 0;
		stopCount = 0;
		for(i=0;i<POND_DEPTH;++i) {
			inst = (cell->genome[wordPtr] >> shiftPtr) & 0xf;
			/* Four STOP instructions in a row is considered the end.
			 * The probability of this being wrong is *very* small, and
			 * could only occur if you had four STOPs in a row inside
			 * a LOOP/REP pair that's always false. In any case, this
			 * would always result in our *underestimating* the size of
			 * the genome and would never result in an overestimation. */
            fprintf(file, "%x", inst);
			if (inst == 0xf) { /* STOP */
				if (++stopCount >= 4)
                    fprintf(file, "%s", "stopped");
					break;
			} else stopCount = 0;
			if ((shiftPtr += 4) >= SYSWORD_BITS) {
				if (++wordPtr >= POND_DEPTH_SYSWORDS) {
					wordPtr = EXEC_START_WORD;
					shiftPtr = EXEC_START_BIT;
				} else shiftPtr = 0;
			}
		}
	fprintf(file,"\n");
}
int main(int argc, char** argv) {
    FILE *file = fopen("file.txt", "r");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return 1;
    }

    for (int i = 0; i < POND_SIZE_X; i++) {
        for (int j = 0; j < POND_SIZE_Y; j++) {
            char line[GENOME_SIZE];
            if (fgets(line, sizeof(line), file) == NULL) {
                printf("Failed to read the file.\n");
                return 1;
            }
            //printf("%s\n\n", line); 
            pond[i][j] = readCell(line);
           /* while (!strchr(line, '\n')) {
                printf("%s", "haha");
                break;
                }*/
             }
        }
    

    
    // Close the file
    fclose(file);

    FILE *file1 = fopen("file1.txt", "w");
    if (file1 == NULL) {
        printf("Failed to create the file.\n");
        return 1;
    }
    printUnpackedCell(pond[0][0]);
    
    //writeCell(file1, &c1);
    fclose(file1);
    return 0;
    }

