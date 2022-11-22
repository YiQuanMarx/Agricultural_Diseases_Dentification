：loop
	python mkdata.py
	python pa.py < getIn.in > out1.out
	python bruteF.py < getIn.in > out2.out
	fc out1.out out2.out
if not errorlevel 1 goto loop
	pause
goto loop