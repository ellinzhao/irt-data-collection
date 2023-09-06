moving = imread('align_1018.png');
fixed = imread('ref_1018.png');
[movingPoints, fixedPoints] = cpselect(moving, fixed, 'Wait', true);

tform = fitgeotform2d(movingPoints, fixedPoints, 'similarity');
Jregistered = imwarp(moving, tform, OutputView=imref2d(size(fixed)));
imshowpair(fixed, Jregistered);
