#include <stdio.h>
#include <stdlib.h>

#include "processor.h"
#include "disassemble.h"

int seg(int a){
    int signext = (a >> 15 == 0)? a : (a | 0xffff0000);
    return signext;
}

void execute_one_inst(processor_t* p, int prompt, int print_regs)
{
  inst_t inst;
  reg_t tmp;
  /* fetch an instruction */
  inst.bits = load_mem(p->pc, SIZE_WORD);

  /* interactive-mode prompt */
  if(prompt)
  {
    if (prompt == 1) {
      printf("simulator paused, enter to continue...");
      while(getchar() != '\n')
        ;
    }
    printf("%08x: ",p->pc);
    disassemble(inst);
  }

  switch (inst.rtype.opcode) /* could also use e.g. inst.itype.opcode */
  {
  case 0x0: // opcode == 0x0 (SPECIAL)

    switch (inst.rtype.funct)
    {
    case 0x0:
      p->R[inst.rtype.rd] = p->R[inst.rtype.rt] << inst.rtype.shamt;
      p->pc += 4;
      break;
            
    case 0x2:
      p->R[inst.rtype.rd] = (unsigned int)p->R[inst.rtype.rt] >> inst.rtype.shamt;
      p->pc += 4;
      break;

    case 0x3:
      p->R[inst.rtype.rd] = (signed int)p->R[inst.rtype.rt] >> inst.rtype.shamt;
      p->pc += 4;
      break;
        
    case 0x8:
      p->pc = p->R[inst.rtype.rs];
//      p->pc +=4;
      break;//
      
    case 0x9:
      tmp = p->pc + 4;
      p->pc = p->R[inst.rtype.rs];
      p->R[inst.rtype.rd] = tmp;
      break;
      
    case 0xc: // funct == 0xc (SYSCALL)
      handle_syscall(p);
      p->pc += 4;
      break;

    case 0x21:
      p->R[inst.rtype.rd] = p->R[inst.rtype.rs] + p->R[inst.rtype.rt];
      p->pc += 4;
      break;
      
    case 0x23:
      p->R[inst.rtype.rd] = p->R[inst.rtype.rs] - p->R[inst.rtype.rt];
      p->pc += 4;
      break;
            
    case 0x24:
      p->R[inst.rtype.rd] = p->R[inst.rtype.rs] & p->R[inst.rtype.rt];
      p->pc += 4;
      break;
            
    case 0x25: // funct == 0x25 (OR)
      p->R[inst.rtype.rd] = p->R[inst.rtype.rs] | p->R[inst.rtype.rt];
      p->pc += 4;
      break;

    case 0x26:
      p->R[inst.rtype.rd] = p->R[inst.rtype.rs] ^ p->R[inst.rtype.rt];
      p->pc += 4;
      break;
            
    case 0x27:
      p->R[inst.rtype.rd] = ~(p->R[inst.rtype.rs] | p->R[inst.rtype.rt]);
      p->pc += 4;
      break;
            
    case 0x2a:
      p->R[inst.rtype.rd] = (signed int)p->R[inst.rtype.rs] < (signed int)p->R[inst.rtype.rt];
      p->pc += 4;
      break;
            
    case 0x2b:
      p->R[inst.rtype.rd] = p->R[inst.rtype.rs] < p->R[inst.rtype.rt];
      p->pc += 4;
      break;
            
    default: // undefined funct
      fprintf(stderr, "%s: pc=%08x, illegal instruction=%08x\n", __FUNCTION__, p->pc, inst.bits);
      exit(-1);
      break;
    }
    break;

          
  case 0x4:
    if (p->R[inst.itype.rs] == p->R[inst.itype.rt]) {
        p->pc = p->pc + 4 + 4 * seg(inst.itype.imm);
    }else{
    p->pc+=4;
    }
    break;

          
  case 0x5:
    if (p->R[inst.itype.rs] != p->R[inst.itype.rt]) {
        p->pc = p->pc + 4 + 4 * seg(inst.itype.imm);//
    }else{
    p->pc += 4;
    }
    break;
          
  case 0x9:
    p->R[inst.itype.rt] = p->R[inst.itype.rs] + seg(inst.itype.imm);//
    p->pc += 4;
    break;
  
  case 0xa:
    p->R[inst.itype.rt] = (signed int)p->R[inst.itype.rs] < seg(inst.itype.imm);//
    p->pc += 4;
    break;
          
  case 0xb:
    p->R[inst.itype.rt] = p->R[inst.itype.rs] < seg(inst.itype.imm);//
    p->pc += 4;
    break;
          
  case 0xc:
    p->R[inst.itype.rt] = p->R[inst.itype.rs] & inst.itype.imm;
    p->pc += 4;
    break;
          
  case 0xd: // opcode == 0xd (ORI)
    p->R[inst.itype.rt] = p->R[inst.itype.rs] | inst.itype.imm;
    p->pc += 4;
    break;

  case 0xe:
    p->R[inst.itype.rt] = p->R[inst.itype.rs] ^ inst.itype.imm;
    p->pc += 4;
    break;
  
  case 0xf:
    p->R[inst.itype.rt] = inst.itype.imm << 16;
    p->pc += 4;
    break;
    
  case 0x20:
    p->R[inst.itype.rt] = seg(load_mem(p->R[inst.itype.rs] + seg(inst.itype.imm), SIZE_BYTE));
    p->pc += 4;
    break;//
    
  case 0x21:
    p->R[inst.itype.rt] = seg(load_mem(p->R[inst.itype.rs] + seg(inst.itype.imm), SIZE_HALF_WORD));
    p->pc += 4;
    break;//

  case 0x23:
    p->R[inst.itype.rt] = load_mem(p->R[inst.itype.rs] + seg(inst.itype.imm), SIZE_WORD);
    p->pc += 4;
    break;//
    
  case 0x24:
    p->R[inst.itype.rt] = load_mem(p->R[inst.itype.rs] + seg(inst.itype.imm), SIZE_BYTE);
    p->pc += 4;
    break;//
   
  case 0x25:
    p->R[inst.itype.rt] = load_mem(p->R[inst.itype.rs] + seg(inst.itype.imm), SIZE_HALF_WORD);
    p->pc += 4;
    break;//
   
  case 0x28:
    store_mem(p->R[inst.itype.rs] + seg(inst.itype.imm), SIZE_BYTE, p->R[inst.itype.rt]);
    p->pc += 4;
    break;//
          
  case 0x29:
    store_mem(p->R[inst.itype.rs] + seg(inst.itype.imm), SIZE_HALF_WORD, p->R[inst.itype.rt]);
    p->pc += 4;
    break;//
          
  case 0x2b:
    store_mem(p->R[inst.itype.rs] + seg(inst.itype.imm), SIZE_WORD, p->R[inst.itype.rt]);
    p->pc += 4;
    break;//
          
  case 0x2: // opcode == 0x2 (J)
    p->pc = ((p->pc+4) & 0xf0000000) | (inst.jtype.addr << 2);
    break;
          
  case 0x3:
    p->R[31] = p->pc+4;
    p->pc = ((p->pc+4) & 0xf0000000) | (inst.jtype.addr << 2);
    break;
          
  default: // undefined opcode
    fprintf(stderr, "%s: pc=%08x, illegal instruction: %08x\n", __FUNCTION__, p->pc, inst.bits);
    exit(-1);
    break;
  }

  // enforce $0 being hard-wired to 0
  p->R[0] = 0;

  if(print_regs)
    print_registers(p);
}

void init_processor(processor_t* p)
{
  int i;

  /* initialize pc to 0x1000 */
  p->pc = 0x1000;

  /* zero out all registers */
  for (i=0; i<32; i++)
  {
    p->R[i] = 0;
  }
}

void print_registers(processor_t* p)
{
  int i,j;
  for (i=0; i<8; i++)
  {
    for (j=0; j<4; j++)
      printf("r%2d=%08x ",i*4+j,p->R[i*4+j]);
    puts("");
  }
}

void handle_syscall(processor_t* p)
{
  reg_t i;

  switch (p->R[2]) // syscall number is given by $v0 ($2)
  {
  case 1: // print an integer
    printf("%d", p->R[4]);
    break;

  case 4: // print a string
    for(i = p->R[4]; i < MEM_SIZE && load_mem(i, SIZE_BYTE); i++)
      printf("%c", load_mem(i, SIZE_BYTE));
    break;

  case 10: // exit
    printf("exiting the simulator\n");
    exit(0);
    break;

  case 11: // print a character
    printf("%c", p->R[4]);
    break;

  default: // undefined syscall
    fprintf(stderr, "%s: illegal syscall number %d\n", __FUNCTION__, p->R[2]);
    exit(-1);
    break;
  }
}
