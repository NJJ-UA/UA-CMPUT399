function [x,y] = Stage1Detector( I, W )

% assumes a 24-by-24 bounding box
wxsz=24;
wysz=24;

% compute score function by correlating I with filter W
S = imfilter(I,W,'same','corr','replicate');

% do non-maximum suppression
Sdilated = imdilate(S,strel('disk',12));
B = Sdilated==S;
[y,x] = find(B);

% draw bounding boxes on the image
% bbox=[x(:)-wxsz/2 y(:)-wysz/2 x(:)+wxsz/2 y(:)+wysz/2];
% clf, showimage(I), showbbox(bbox);

end
