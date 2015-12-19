function [x,y,score] = Stage2Detector(I,x,y, W,b,cellsize,threshold)


hogsize=24/cellsize;
[m,n]=size(I);
length=24;

npoints=size(x,1);
totalhog=zeros(npoints,hogsize*hogsize*31);
for i=1:npoints
    ymin=y(i)-length/2;
    ymax=y(i)+length/2;
    xmin=x(i)-length/2;
    xmax=x(i)+length/2;

    
    if xmin<1 || ymin<1 || xmax>n||ymax>m
        continue
    end
    img=I(ymin:ymax,xmin:xmax);
    hog=vl_hog(im2single(img),cellsize);
    hog=reshape(hog,1,hogsize*hogsize*31);
    totalhog(i,:)=hog;
    
end

score=totalhog*W+b;
x=x(score>threshold);
y=y(score>threshold);
score=score(score>threshold);
end
