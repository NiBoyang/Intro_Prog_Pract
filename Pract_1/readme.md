## Requirements

Practical Exercise 1 Worth 25 marks (Your total marks from all three exercises are your course mark)
In this exercise I want you to read in a simple file of atomic coordinates in this format
C5' 40.228 17.831 13.474
O5' 40.761 17.421 12.230
C4' 41.034 18.978 14.030
O4' 41.042 20.083 13.138
C3' 40.382 19.568 15.268
O3' 40.575 18.803 16.426
C2' 40.973 20.967 15.342
O2' 42.117 20.993 16.172
C1' 41.343 21.265 13.892

I’d like you to do several things with this data
1.	Find the maximum and minimum coordinates of a box that encompasses all the coordinates
2.	Find the simple centre of gravity of the coordinates in the file
3.	Print out which atoms are bonded to which (eg C4’ is bonded to O5’) To simplify this you can assume that any two atoms that are closer than 1.8 A (the file is scaled in Angstroms) are bonded.
Successfully completing this will gain a pass mark in this practical (13 Marks) to gain an enhanced mark, you need to add comments to the code that demonstrate your understanding, add a readme file that explains (to a user on a different computer for instance) how to use the code, and add code that enhances the functionality and makes it more useful (eg detects error conditions, adds a more complex analysis of the data). I have provided you with two test files, one with mistakes in the input, they are
~prt/intro_prog_pract/pract1/testdata.dat .
~prt/intro_prog_pract/pract1/testdata_bad_item.dat .

These files are available in the directory /datastore/home/prt/intro_prog_pract/pract1 (also known as ~prt/intro_prog_pract/pract1 your work should go in your own pract1 directory so if your username was s187654, then your home directory would be /datastore/home/s197654. In there you should already have a directory intro_prog_pract, you get into that directory after you have logged in by typing
cd intro_prog_pract
you then make a directory called pract1
mkdir pract1
then change into that directory
cd pract1

From this point if you log in and want to get to the pract1 directory do
cd intro_prog_pract/pract1

to copy the test files from me do
cp ~prt/intro_prog_pract/pract1/testdata* .
These means copy all the files that match the pattern ~prt/intro_prog_pract/pract1/testdata* into the current directory.
I will walk through the creation of a basic program in the practical video
![image](https://user-images.githubusercontent.com/87428829/146456942-c857b4bf-673e-4b90-aea2-4e7bfc49e67f.png)