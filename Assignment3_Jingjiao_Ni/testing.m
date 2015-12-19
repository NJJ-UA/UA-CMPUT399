close all;
clear all;

cellsize=4;

level=55;
factor=1.05;
start=35;
threshold=[2,1.7,1.5,1.8,2,2,2,2,1,2,2,2,2,2,1,1,1,1,1,1.2,2];

load('classifiers.mat');

OI=imread('test_img.jpg');

I=double(rgb2gray(OI));
I=meanvarpatchnorm(I);

GI = GaussianPyramid(I,level,factor);

bbox1=[];
bbox2=[];
for i=start:level
    img=GI{i};
    [x,y] = Stage1Detector( double(img), reshape(Wbestrp,24,24));
    
    length=24*(factor^(i-1));
    xshow=x*(factor^(i-1));
    yshow=y*(factor^(i-1));
    bbox=[xshow(:)-length/2 yshow(:)-length/2 xshow(:)+length/2 yshow(:)+length/2];
    bbox1=[bbox1;bbox];
    %figure(1),showimage(OI), showbbox(bbox);
    
    
    [x,y,score] = Stage2Detector( double(img),x,y, Wbesthog,bbesthog,cellsize,threshold(i-start+1));
    
    length=24*(factor^(i-1));
    xshow=x*(factor^(i-1));
    yshow=y*(factor^(i-1));
    bbox=[xshow(:)-length/2 yshow(:)-length/2 xshow(:)+length/2 yshow(:)+length/2 xshow yshow score];
    bbox2=[bbox2;bbox];
    %figure(2),showimage(OI), showbbox(bbox(:,1:4));
end


bbox2=removerepeat(bbox2);
bbox2=bbox2(:,1:4);
figure(1),showimage(OI), showbbox(bbox1,[1,1,0]);
figure(2),showimage(OI), showbbox(bbox2,[0,1,0]);
hold off