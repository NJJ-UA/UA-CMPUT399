function bboxout=removerepeat(bboxin)
    faces={};
    nface=0;
    bboxout=[];
    for i=1:size(bboxin,1)
        if nface==0;
            nface=nface+1;
            faces{nface}=[bboxin(i,:)];
        else
            newface=true;
            for j=1:nface
                d=pdist([mean(faces{j}(:,5:6),1) ;bboxin(i,5:6),],'euclidean');
                if d<18
                    faces{j}=[faces{j} ;bboxin(i,:)];
                    newface=false;
                    break
                end
            end
            if newface==true
                nface=nface+1;
                faces{nface}=[bboxin(i,:)];
            end
        end
    end
    
    for i=1:nface
        if size(faces{i},1)==1
            bboxout=[bboxout ; faces{i}];
        else
            [~,I]=max(faces{i});
            bboxout=[bboxout ; faces{i}(I(:,7),:)];
        end 
    end
end