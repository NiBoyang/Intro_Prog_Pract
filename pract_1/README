File name: pract1.py

=========
Function:
=========
  pract1.py is a program that will read a file formated as

  ---------------
  Atom_name X Y Z
     ***    * * *
  ......
  ---------------

  Then the program will calculate the maximum and minimum extent of the molecule,simple
  centre of gravity, weighted centre of gravity, and bonding between atoms if applicable.

  Furthermore, it will show you a simple graph of binding positions with annotation on
  the points.

  Exceptions:
        1. If there exists invalid floating numbers in your data file, like 'x.xxx', the
        program will tell you where is it and then exits. There will be no result output.

        2. If there exists invalid element symbol, like 'mg', the program will tell you
        where is it. The only output will be the simple centre of gravity. Weighted centre
        and bondings will not appear due to incorrect element.

======
Usage:
======
  Put this python file and your data file in the same directory, then type in terminal:
    python pract1.py input_file  *or*  ./pract1.py input_file

===========
Limitation:
===========
 1. If there are many incorrect inputs in your data file, you can only deal with them one by one,
  because the program only tells you one error each run.

 2. Since the program gets atomic mass from an intact periodic table, so if you have input an incorrect
  atom symbol which is a valid element, the program cannot detect it. For instance, you input Pb instead
  of Mg, the program can still run and give you the result. Only symbol not in periodic table will be
  detected, like PB instead of Pb.

 3. Call Axes3D from matplotlib will lead to TWO LINES OF 'libGL errors' at the end of the output, but it will not
  influence the output of the binding graph.
