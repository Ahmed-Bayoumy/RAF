
Include "airfoil.geo";
//+

ymax = 4;
xmax = 10;
n_inlet = 60;
n_vertical = 90;
r_vertical = 1/0.95;
n_airfoil = 50;
n_wake = 100;
r_wake = 1/0.95;
n_trailing = 4;
r_trailing = 1;
c1[] = Point{128};

//+
Point(129) = {-0.5, ymax, 0, 1.0};
//+
Point(130) = {-0.5, -ymax, 0, 1.0};
//+
Point(131) = {1, ymax, 0, 1.0};
//+
Point(132) = {1, -ymax, 0, 1.0};
//+
Point(133) = {xmax, ymax, 0, 1.0};
//+
Point(134) = {xmax, -ymax, 0, 1.0};
//+
Point(135) = {xmax, 0, 0, 1.0};
//+
Point(136) = {xmax, c1[1], 0, 1.0};
//+
Circle(2) = {130, 64, 129};
//+
Line(3) = {57, 129};
//+
Line(4) = {71, 130};
//+
Line(5) = {129, 131};
//+
Line(6) = {130, 132};
//+
Line(7) = {131, 133};
//+
Line(8) = {132, 134};
//+
Line(9) = {136, 134};
//+
Line(10) = {135, 133};
//+
Line(11) = {1, 131};
//+
Line(12) = {128, 132};
//+
Line(13) = {128, 136};
//+
Line(14) = {1, 135};
//+
Line(15) = {135, 136};

//+
Split Curve {1} Point {57, 71};
//+
Split Curve {15} Point {128, 1};
//+
Split Curve {16} Point {1, 128};
//+
Split Curve {17} Point {1, 128};
//+
Transfinite Curve {2, 19} = n_inlet Using Progression 1;


//+
Transfinite Curve {3, 11, 10, 4, 12, 9} = n_vertical Using Progression r_vertical;
//+
Transfinite Curve {5, 22} = n_airfoil Using Bump 2;
//+
Transfinite Curve {5, 22, 20, 6} = n_airfoil Using Progression 1;
//+
Transfinite Curve {20, 6} = n_airfoil Using Bump 2;
//+
Transfinite Curve {13, 14} = n_wake Using Progression r_wake;
//+
Transfinite Curve {7, 8} = n_wake Using Bump 0.2;
//+
Transfinite Curve {21, 18} = n_trailing Using Progression r_trailing;
//+
Curve Loop(1) = {2, -3, 19, 4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {3, 5, -11, 22};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {4, 6, -12, -20};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {11, 7, -10, -14};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {12, 8, -9, -13};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {21, 14, 18, -13};
//+
Plane Surface(6) = {6};
//+
Transfinite Surface {1};
//+
Transfinite Surface {2};
//+
Transfinite Surface {4};
//+
Transfinite Surface {6};
//+
Transfinite Surface {5};
//+
Transfinite Surface {3};
//+
Recombine Surface {1, 2, 4, 6, 5, 3};
//+
Physical Surface("Fluid", 23) = {1, 2, 4, 6, 5, 3};
//+
Physical Curve("airfoil", 24) = {19, 22, 21, 20};
//+
Physical Curve("farfield", 25) = {2, 5, 7, 10, 18, 9, 8, 6};
