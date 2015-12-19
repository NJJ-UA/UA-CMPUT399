function [jaccardScore,bestThresh]=convertToBinary(match_fracs)
% computes and displays a series binary images for 'match_fracs' 
% corresponding to several thresholds varying from 0 to 1.
warning('off','all');
figure;
load('Lip6OutdoorDataSet/Lip6OutdoorGroundTruth.mat');

results=[];
for thresh=0:0.01:0.4
    matches=zeros(size(match_fracs));
    matches(match_fracs>=thresh)=1;
    js=sum(sum(and(matches,truth))) / sum(sum(or(matches,truth)));
    results=[results [thresh;js]];
    filename=sprintf('matches_thresh_%6.4f.jpg', thresh);
    title(filename);
    imshow(matches);    
    imwrite(matches,filename)
    pause(0.2);
end

[jaccardScore,index]=max(results(2,:));
bestThresh=results(1,index);

end
