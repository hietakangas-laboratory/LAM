DESCRIPTION:
-----------
The run_split_count.bat is a batch-file example of how to perform LAM's Count-functionality
consecutively to multiple datasets resulting from split_dataset.py. The batch file can be
executed from console, and completes without additional user-input if given with argument -D.
After the batch-file has executed, the datasets would be ready for re-combining with the
companion script combineSets.py.


COMMAND BREAKDOWN:
-----------------
lam-run -p E:\ALLSTAT_split\R1R2 -o c -b 9 -H 0 -BWGM -D


lam-run  :  launch LAM
	    Functions only if installed through setup.py. Else launch command must be
	    python <path\to\LAM-master\src\run.py>

-p PATH  :  Path to analysis directory

-o c     :  Primary functionalities to perform, here "c" == Count

-b 9     :  The number of bins for the analysis

-H 0 	 :  Data header row set to zero

-BWGM	 :  Toggle LAM-functionalities to False,
	    B == border detection
	    W == width approximation
	    G == GUI
	    M == measurement point, MP

-D 	 :  Force LAM not to interact with user