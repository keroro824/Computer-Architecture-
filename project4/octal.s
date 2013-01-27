OCTAL:
		andi $r0 $r0 0
		lw $r1 0($r0)		
		addi $r0 $r0 6
		srlv $r1 $r1 $r0
		andi $r2 $r1 0x7	
		andi $r0 $r0 0
		addi $r0 $r0 4
		sllv $r2 $r2 $r0
		andi $r0 $r0 0
		lw $r1 0($r0)		
		addi $r0 $r0 3
		srlv $r1 $r1 $r0
		andi $r1 $r1 0x7
		or $r2 $r2 $r1		
		andi $r0 $r0 0
		addi $r0 $r0 4
		sllv $r2 $r2 $r0
		andi $r0 $r0 0
		lw $r1 0($r0)
		andi $r1 $r1 0x7
		or $r2 $r2 $r1
		disp $r2, 0
		jr $r3
		