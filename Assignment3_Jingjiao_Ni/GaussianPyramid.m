function GI = GaussianPyramid(I,level,factor)
% return a cell array
GI = cell(level,1);
GI{1} = imfilter(I,fspecial('Gaussian',[7 7],1),'same','conv','replicate');
for n=2:level,
    h = round(size(GI{n-1},1)/factor);
    w = round(size(GI{n-1},2)/factor);
    GI{n} = imresize(GI{n-1},[h w],'bicubic','Antialiasing',true);
end