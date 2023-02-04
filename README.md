# crt-convergence-analyzer
A CRT Convergence Analyzing Tool based on OpenCV

This a simple proof-of-concept program that demonstrates the same functionality as advanced CRT convergence tools from back in the day. To do this, it follows a few steps:
1. Identify dots from a dot test pattern
2. Crop and scale/interpolate each individual dot
3. Look for areas with excessive red and blue contours
4. Calculate scaled vector to represent severity of mis-calibration
5. Take all data and analyze for global patterns (Still in Progress)

The tool is simple to use and only requires a path and scale factor (0-100). It requires that the OpenCV library is installed.

python3 crtconvanalysis Path_to_image scale

To run this with the example photo included (of a BVM-A24E1WU) copy and run this command:

python3 crtconvanalysis.py A24-bad-convergence.jpg 10
