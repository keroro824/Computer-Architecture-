StringLen:
			andi $r0 $r0 0 
loop:	
			lw $r1 0($r0)
			zb $r2 $r1
			sw $r3 0($r0)
			andi $r3 $r3 0
			bne $r2 $r3 finish1
			lw $r3 0($r0)
			addi $r0 $r0 1
			j loop
finish1:
            ffo $r2 $r1
			ori $r3 $r3 8
			slt $r1 $r2 $r3
			andi $r3 $r3 0
			andi $r2 $r2 0
			bne $r1 $r3 another
			j finish2
another:
			lw $r3 0($r0)
			ori $r2 $r2 1
			sllv $r0 $r0 $r2
			add $r0 $r0 $r2
			disp $r0, 0
			jr $r3
finish2:
			lw $r3 0($r0)
			ori $r2 $r2 1
			sllv $r0 $r0 $r2
			disp $r0, 0
			jr $r3
