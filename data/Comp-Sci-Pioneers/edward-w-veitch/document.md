# Edward W. Veitch

## Later reflections on the diagram's design
Veitch wrote about the development of the Veitch diagram and its interpretation:

- The problem is how to depict a Boolean function of n variables so the human eye can easily see how to simplify the function.  
  - A function of four variables has sixteen input combinations and the diagram has sixteen different squares to be filled from the truth table that defines the function.
  - The primary difference between the Veitch and Karnaugh versions is that the Veitch diagram presents the data in the binary sequence used in the truth table while the Karnaugh map interchanges the third and fourth rows and the third and fourth columns.
  - The general digital computer community chose the Karnaugh approach. Veitch accepted this decision, even though in early 1952, before his presentation, he had almost changed to that approach but decided against it. A few years later several textbooks described the K-map, a few of them designating it a Veitch diagram.



### The original Veitch diagram
It was known that one way to represent the function was as points on the corners of an n-dimensional cube. Two adjacent corners such as the two on the upper right could be defined as the upper right corners and the four corners on the front of the cube could be defined as the front corners. For four, five, or six variables the problem becomes more complicated.

Depicting a multi-dimensional cube on a flat diagram that makes it easy to see these relationships:

- For three dimensions, Veitch drew a 2×2 set of squares for the top of the cube and a second set for the bottom of the cube with a small space between the two sets of squares. Within the 2×2 set on the top the simplification groups are any horizontal or vertical pair or all of the four cells. The only adjacencies between the top and bottom sets are a one-to-one connection between each square of the top set and corresponding cell of the bottom set. A similar rule applies to the four variable cases, which is sometimes drawn as a cube inside of another cube with corresponding corners all connected.
- The four variable Veitch diagram would then be four 2×2 sets in a larger square with a small space between each pair of sets. Thus a horizontal pair in the top left set can combine with a matching pair in the bottom left set or with the top right set or possibly with all four sets to make an eight cell group.
- For five variables or six variables the same rule applies. The five variable diagram consists of two four variable diagrams drawn next to each other with a larger space between them. Matches between the two four variable diagrams are between cells that are next to each other when one map is overlaid over the other.


In a last minute change before his presentation Veitch removed the spacing between the 2×2 cell groups. This was a poor decision because it made it more difficult for the user to grasp the overall structure of the function, as well as the rules Veitch used in recognizing simplifications. In his last years before his death in 2013, Veitch learned from solving Sudoku puzzles that spaces or heavy lines between groups of boxes can be very helpful especially if one has poor eyesight, such as Veitch had.