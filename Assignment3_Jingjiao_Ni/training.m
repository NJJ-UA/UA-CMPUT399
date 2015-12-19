close all;
clear all;


cellsize=4;

setup;

load('possamples.mat')
load('negsamples.mat')
 

% possamples=possamples(:,:,1:1000);
% negsamples=negsamples(:,:,1:5000);

npos=size(possamples,3);
nneg=size(negsamples,3); 

% posrand=randperm(npos);
% negrand=randperm(nneg);
% possamples=possamples(:,:,posrand);
% negsamples=negsamples(;,;,negrand);

possamples=double(possamples);
negsamples=double(negsamples);
fprintf('load  %6d positive samples \n',npos)
fprintf('load  %6d negative samples \n\n',nneg)

possamples=meanvarpatchnorm(possamples);
negsamples=meanvarpatchnorm(negsamples);



xsz=size(possamples,2);
ysz=size(possamples,1);
Xpos=transpose(reshape(possamples,ysz*xsz,npos));
Xneg=transpose(reshape(negsamples,ysz*xsz,nneg));




ypos=ones(npos,1);
yneg=-ones(nneg,1);

Xtotal=[Xpos ; Xneg];
ytotal=[ypos; yneg];


% ntrainpos=1000;
% ntrainneg=1000;
% indpostrain=1:ntrainpos; indposval=indpostrain+ntrainpos;
% indnegtrain=1:ntrainneg; indnegval=indnegtrain+ntrainneg;

ntrainpos=0.9*npos;
% ntrainneg=0.9*nneg;
ntrainneg=ntrainpos*4;

indpostrain=1:ntrainpos; 
indposval=setdiff(1:npos,indpostrain);
indnegtrain=1:ntrainneg;
indnegval=setdiff(1:nneg,indnegtrain);

Xtrain=[Xpos(indpostrain,:); Xneg(indnegtrain,:)];
ytrain=[ypos(indpostrain); yneg(indnegtrain)];
Xval=[Xpos(indposval,:); Xneg(indnegval,:)];
yval=[ypos(indposval); yneg(indnegval)];


epsilon = .000001;
kerneloption= 1; % degree of polynomial kernel (1=linear)
kernel='poly';   % polynomial kernel
verbose = 0;


%Call=[0.00001 0.0001 0.0010.01 0.1 1 10 100];
Call=[0.00001 0.0001 0.001 0.01];
accbest=-inf; 
for i=1:length(Call)
  tic
  C=Call(i);

  [Xsup,yalpha,b,~]=svmclass(Xtrain,ytrain,C,epsilon,kernel,kerneloption,verbose);
  [~,acctrain,~]=svmvalmod(Xtrain,ytrain,Xsup,yalpha,b,kernel,kerneloption);
  [~,accval,~]=svmvalmod(Xval,yval,Xsup,yalpha,b,kernel,kerneloption);
  W = (yalpha'*Xsup)';

  fprintf('C=%1.5f | Training accuracy: %1.3f; validation accuracy: %1.3f \n',C,acctrain,accval);
  
  if accbest<accval,
      accbest = accval;
      Cbest = C;
      Wbest = W;
      bbest = b;
  end
  toc
end
fprintf(' -> Best accuracy %1.3f for C=%1.5f\n',accbest,Cbest)




[Xsup,yalpha,b,~]=svmclass(Xtotal,ytotal,Cbest,epsilon,kernel,kerneloption,verbose);
Wbestrp=(yalpha'*Xsup)';
bbestrp=b;
[~,acctotal,~]=svmvalmod(Xtotal,ytotal,Xsup,yalpha,b,kernel,kerneloption);
fprintf('C=%1.5f | Total accuracy: %1.3f; \n',Cbest,acctotal);


conftrainnew = Xtrain*Wbestrp+bbestrp;
train2logic=(conftrainnew>0)*2-1~=ytrain;
train2logic=train2logic|(conftrainnew>0);
Xtrain2 = Xtrain(train2logic,:);

confvalnew = Xval*Wbestrp+bbestrp;
val2logic=(confvalnew>0)*2-1~=yval;
val2logic=val2logic|(confvalnew>0);
Xval2 = Xval(val2logic,:);

Xtrain2=Xtrain2';
Xval2=Xval2';

ntrain2=size(Xtrain2,2);
nval2=size(Xval2,2);
ytrain2=ytrain(train2logic,:);
yval2=yval(val2logic);

Xtrain2=reshape(Xtrain2,ysz,xsz,ntrain2);
Xval2=reshape(Xval2,ysz,xsz,nval2);

yszhog=ysz/cellsize;
xszhog=xsz/cellsize;
Xtrainhog=zeros(ntrain2,yszhog*xszhog*31);
Xvalhog=zeros(nval2,yszhog*xszhog*31);
for i=1:ntrain2
    hog=vl_hog(im2single(Xtrain2(:,:,i)),cellsize);
    hog=reshape(hog,1,yszhog*xszhog*31);
    Xtrainhog(i,:)=hog;
end
for i=1:nval2
    hog=vl_hog(im2single(Xval2(:,:,i)),cellsize);
    hog=reshape(hog,1,yszhog*xszhog*31);
    Xvalhog(i,:)=hog;
end
    
Call=[0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000];
accbest=-inf; 

for i=1:length(Call)
  tic
  C=Call(i);
 
  [Xsup,yalpha,b,~]=svmclass(Xtrainhog,ytrain2,C,epsilon,kernel,kerneloption,verbose);
  [~,acctrain,~]=svmvalmod(Xtrainhog,ytrain2,Xsup,yalpha,b,kernel,kerneloption);
  [~,accval,~]=svmvalmod(Xvalhog,yval2,Xsup,yalpha,b,kernel,kerneloption);
  W = (yalpha'*Xsup)';
  
  fprintf('C=%1.5f | Training accuracy: %1.3f; validation accuracy: %1.3f \n',C,acctrain,accval);
  
  if accbest<accval,
      accbest = accval;
      Cbest = C;
      Wbest = W;
      bbest = b;
  end
  toc
end
fprintf(' -> Best accuracy %1.3f for C=%1.5f\n',accbest,Cbest)


conftotalnew=Xtotal*Wbestrp+bbestrp;
totallogic=(conftotalnew>0)*2-1~=ytotal;
totallogic=totallogic|(conftotalnew>0);
Xtotal=cat(3,possamples,negsamples);
Xtotal2 = Xtotal(:,:,totallogic);
ytotal2=ytotal(totallogic,:);

ntotalhog=size(Xtotal2,3);
Xtotalhog=zeros(ntotalhog,yszhog*xszhog*31);

for i=1:ntotalhog
    hog=vl_hog(im2single(Xtotal2(:,:,i)),cellsize);
    hog=reshape(hog,1,yszhog*xszhog*31);
    Xtotalhog(i,:)=hog;
end
[Xsup,yalpha,b,~]=svmclass(Xtotalhog,ytotal2,Cbest,epsilon,kernel,kerneloption,verbose);
Wbesthog=(yalpha'*Xsup)';
bbesthog=b;

[~,acctotalhog,~]=svmvalmod(Xtotalhog,ytotal2,Xsup,yalpha,b,kernel,kerneloption);
fprintf('C=%1.5f | Total accuracy: %1.3f; \n',Cbest,acctotalhog);

save('classifiers.mat','Wbestrp','bbestrp','Wbesthog','bbesthog');
