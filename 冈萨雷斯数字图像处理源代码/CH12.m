%% 第十二章 对象识别

%% 例12.1 gscale dftcorr 使用相关来匹配图像
clc
clear
f = imread('..\Pictures\images_ch12\Fig1201(a)(Hurricane Andrew).tif');
imshowMy(f)
title('原始图像')

w = imread('..\Pictures\images_ch12\Fig1201(b)(hurricane_mask).tif');
imshowMy(w)
title('模板图像')

g = dftcorr(f,w);
gs = gscale(g);
imshowMy(gs)

[I, J] = find(g == max(g(:)))

imshowMy(gs > 254)


%% 例12.2 多谱数据的贝叶斯分类
clc
clear

f1 = imread('..\Pictures\images_ch11\Fig1125(a)(WashingtonDC_Band1_512).tif');
f2 = imread('..\Pictures\images_ch11\Fig1125(b)(WashingtonDC_Band2_512).tif');
f3 = imread('..\Pictures\images_ch11\Fig1125(c)(WashingtonDC_Band3_512).tif');
f4 = imread('..\Pictures\images_ch11\Fig1125(d)(WashingtonDC_Band4_512).tif');

B1 = roipoly(f1);
B2 = roipoly(f1);
B3 = roipoly(f1);

stack = cat(3,f1,f2,f3,f4);

[X1,R1] = imstack2vectors(stack, B1);
[X2,R2] = imstack2vectors(stack, B2);
[X3,R3] = imstack2vectors(stack, B3);

Y1 = X1(1:2:end,:);
Y2 = X2(1:2:end,:);
Y3 = X3(1:2:end,:);

[C1,m1] = covmatrix(Y1);
[C2,m2] = covmatrix(Y2);
[C3,m3] = covmatrix(Y3);

CA = cat(3,C1,C2,C3);
MA = cat(2,m1,m2,m3);

dY{1} = bayesgauss(Y1,CA,MA);
dY{2} = bayesgauss(Y2,CA,MA);
dY{3} = bayesgauss(Y3,CA,MA);

IY{1} = find(dY{1} ~=1 );
IY{2} = find(dY{2} ~=2 );
IY{3} = find(dY{3} ~=3 );

%------------------------------------ 训练模式
QQ_training = zeros(3,5);

for i = 1:3
    QQ_training(i,4) = length(dY{i});
    for j = 1:3
       QQ_training(i,j) = length(find(dY{i} == j ));
    end
    QQ_training(i,5) = 100*(1 - length(IY{i})/length(dY{i}));
end
QQ_training

%------------------------------------ 独立模式
Y1 = X1(2:2:end,:);
Y2 = X2(2:2:end,:);
Y3 = X3(2:2:end,:);

dY{1} = bayesgauss(Y1,CA,MA);
dY{2} = bayesgauss(Y2,CA,MA);
dY{3} = bayesgauss(Y3,CA,MA);

IY{1} = find(dY{1} ~=1 );
IY{2} = find(dY{2} ~=2 );
IY{3} = find(dY{3} ~=3 );

QQ_test = zeros(3,5);

for i = 1:3
    QQ_test(i,4) = length(dY{i}); % 样本总数
    for j = 1:3
       QQ_test(i,j) = length(find(dY{i} == j ));
    end
    QQ_test(i,5) = 100*(1 - length(IY{i})/length(dY{i})); % 正确率
end
QQ_test

%% 例12.2 多谱数据的贝叶斯分类(显示了错误点)
clc
clear

f1 = imread('..\Pictures\images_ch11\Fig1125(a)(WashingtonDC_Band1_512).tif');
f2 = imread('..\Pictures\images_ch11\Fig1125(b)(WashingtonDC_Band2_512).tif');
f3 = imread('..\Pictures\images_ch11\Fig1125(c)(WashingtonDC_Band3_512).tif');
f4 = imread('..\Pictures\images_ch11\Fig1125(d)(WashingtonDC_Band4_512).tif');

B1 = roipoly(f1);
B2 = roipoly(f1);
B3 = roipoly(f1);
B = B1|B2|B3;
imshowMy(B)

stack = cat(3,f1,f2,f3,f4);

[X1,R1] = imstack2vectors(stack, B1);
[X2,R2] = imstack2vectors(stack, B2);
[X3,R3] = imstack2vectors(stack, B3);

Y1 = X1(1:2:end,:);
Y2 = X2(1:2:end,:);
Y3 = X3(1:2:end,:);

[C1,m1] = covmatrix(Y1);
[C2,m2] = covmatrix(Y2);
[C3,m3] = covmatrix(Y3);

CA = cat(3,C1,C2,C3);
MA = cat(2,m1,m2,m3);

dY{1} = bayesgauss(Y1,CA,MA);
dY{2} = bayesgauss(Y2,CA,MA);
dY{3} = bayesgauss(Y3,CA,MA);

IY{1} = find(dY{1} ~=1 );
IY{2} = find(dY{2} ~=2 );
IY{3} = find(dY{3} ~=3 );

%------------------------------------ 训练模式
QQ_training = zeros(3,5);

for i = 1:3
    QQ_training(i,4) = length(dY{i});
    for j = 1:3
       QQ_training(i,j) = length(find(dY{i} == j ));
    end
    QQ_training(i,5) = 100*(1 - length(IY{i})/length(dY{i}));
end

% %------------------------------------ 独立模式
% Y1 = X1(2:2:end,:);
% Y2 = X2(2:2:end,:);
% Y3 = X3(2:2:end,:);
% 
% dY{1} = bayesgauss(Y1,CA,MA);
% dY{2} = bayesgauss(Y2,CA,MA);
% dY{3} = bayesgauss(Y3,CA,MA);
% 
% IY{1} = find(dY{1} ~=1 );
% IY{2} = find(dY{2} ~=2 );
% IY{3} = find(dY{3} ~=3 );
% 
% QQ_test = zeros(3,5);
% 
% for i = 1:3
%     QQ_test(i,4) = length(dY{i});
%     for j = 1:3
%        QQ_test(i,j) = length(find(dY{i} == j ));
%     end
%     QQ_test(i,5) = 100*(1 - length(IY{i})/length(dY{i}));
% end

www1 = R1(IY{1},:);
for i=1:length(www1)
    B(www1(i,1),www1(i,2)) = 0;
end

www2 = R2(IY{2},:);
for i=1:length(www2)
    B(www2(i,1),www2(i,2)) = 0;
end

www3 = R3(IY{3},:);
for i=1:length(www3)
    B(www3(i,1),www3(i,2)) = 0;
end

imshowMy(B)

%% 例12.3 基于符号串匹配的对象识别（目前无法运行）
clc
clear

A = imread('..\Pictures\images_ch12\Fig1203(a)(bottle_1).tif');
imshowMy(A)

B = imread('..\Pictures\images_ch12\Fig1203(d)(bottle_2).tif');
imshowMy(B)

b = boundaries(B,4,'cw');
b = b{1};
[M,N] = size(B);
xmin = min(b(:,1));
ymin = min(b(:,2));
[xA,yA] = minperpoly(A,2);
b2 = connectpoly(xA,yA);
B2 = bound2im(b2,M,N,xmin,ymin);
imshowMy(B2)

[xB,yB] = minperpoly(B,2);

[xn,yn] = randvertex(x,y,npix);

angles = polyangles(x,y);
s = floor(angles/45) + 1;
s = int2str(s);
R = strsimilarity(s11,s12);


%% 
clc
clear




%% 
clc
clear




