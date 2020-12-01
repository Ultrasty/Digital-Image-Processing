%% 第九章 形态学图像处理

%% 例9.1 imdilate 膨胀 
clc
clear

A = imread('..\Pictures\images_ch09\Fig0906(a)(broken-text).tif');
info = imfinfo('..\Pictures\images_ch09\Fig0906(a)(broken-text).tif')
B = [0 1 0
     1 1 1
     0 1 0];
 A2 = imdilate(A, B);
 A3 = imdilate(A2, B); % 二次膨胀
 A4 = imdilate(A3, B); % 三次膨胀
%  A2 = imdilate(B,A); % 哈哈...
imshowMy(A)
title('原始图像')

imshowMy(A2)
title('使用结构元素[B]一次膨胀后的图像')

imshowMy(A3)
title('使用结构元素[B]二次膨胀后的图像')

imshowMy(A4)
title('使用结构元素[B]三次膨胀后的图像')

imshowMy(A2-A) % 显示增加的部分
title('使用结构元素[B]一次膨胀后和原图像相比较增加的部分')


%% 例9.3 imerode 
clc
clear

A = imread('..\Pictures\images_ch09\Fig0908(a)(wirebond-mask).tif');
se = strel('disk', 10);
imshowMy(A)
title('原始图像')

A2 = imerode(A, se);
imshowMy(A2)
title('使用结构元素[disk（10）]腐蚀后的图像')

se = strel('disk', 5);
A3 = imerode(A, se);
imshowMy(A3)
title('使用结构元素[disk（5）]腐蚀后的图像')

A4 = imerode(A, strel('disk', 20));
imshowMy(A4)
title('使用结构元素[disk（20）]腐蚀后的图像')

%% 例9.4.1 imopen imclose 注意当结构元素分别使用 square 和 disk 的巨大区别！！！
clc
clear

f = imread('..\Pictures\images_ch09\Fig0910(a)(shapes).tif');
% se = strel('square', 5);  % 结构元素 方型
se = strel('disk', 5);  % 结构元素 圆盘形
imshowMy(f)  % 原始图  图1
title('原始图像')

fo = imopen(f, se);  % 开 图2
imshowMy(fo) 
title('使用结构元素[disk（5）]开操作后的图像')

fc = imclose(f, se); % 闭 图3
imshowMy(fc)
title('使用结构元素[disk（5）]闭操作后的图像')

foc = imclose(fo, se);  % 先开再闭 图4
imshowMy(foc)
title('使用结构元素[disk（5）]先开操作再闭操作后的图像')

fco = imopen(fc, se);   % 先闭再开 图5
imshowMy(fco)  
title('使用结构元素[disk（5）]先闭操作再开操作后的图像')

% 先膨胀再腐蚀 图6
fse = imdilate(f, se);
figure,set(gcf,'outerposition',get(0,'screensize'))
subplot(211),imshow(fse)
title('使用结构元素[disk（5）]先膨胀后的图像')
fes = imerode(fse, se);
subplot(212),imshow(fes)  
title('使用结构元素[disk（5）]先膨胀再腐蚀后的图像')

% 先腐蚀再膨胀 图7
fse = imerode(f, se);
figure, set(gcf,'outerposition',get(0,'screensize'))
subplot(211),imshow(fse)
title('使用结构元素[disk（5）]先腐蚀后的图像')
fes = imdilate(fse, se);
subplot(212),imshow(fes)  
title('使用结构元素[disk（5）]先腐蚀再膨胀后的图像')

%% 例9.4.2 imopen imclose 指纹 
clc
clear

f = imread('..\Pictures\images_ch09\Fig0911(a)(noisy-fingerprint).tif');
se = strel('square', 3);  % 结构元素
% se = strel('disk', 2);  % 结构元素 圆盘形

imshowMy(f)  % 原始图
title('原始图像')

A = imerode(f, se); % 腐蚀
imshowMy(A)
title('使用结构元素[square（3）]腐蚀后的图像')

fo = imopen(f, se);
imshowMy(fo)    % 开
title('使用结构元素[square（3）]开操作后的图像')

fc = imclose(f, se);   % 闭
imshowMy(fc)
title('使用结构元素[square（3）]闭操作后的图像')

foc = imclose(fo, se);  % 先开再闭
imshowMy(foc)
title('使用结构元素[square（3）]先开操作再闭操作后的图像')

fco = imopen(fc, se);   % 先闭再开
imshowMy(fco)
title('使用结构元素[square（3）]先闭操作再开操作后的图像')

%% 例9.5 bwhitmiss 击中或击不中变换
clc
clear

f = imread('..\Pictures\images_ch09\Fig0913(a)(small-squares).tif');
imshowMy(f)
imfinfoMy(f)
title('原始图像')

B1 = strel([0 0 0; 
            0 1 1; 
            0 1 0]); % 击中：要求“击中”所有的“1”位置，不需要考虑“0”位置
B2 = strel([1 1 1; 
            1 0 0; 
            1 0 0]); % 击不中：要求“击不中”所有的“1”位置，不需要考虑“0”位置

B3 = strel([0 1 0; 
            1 1 1; 
            0 1 0]);
B4 = strel([1 0 1; 
            0 0 0; 
            0 0 0]);
        
B5 = strel([0 0 0; 
            0 1 0; 
            0 0 0]); % 击中
B6 = strel([1 1 1; 
            1 0 0; 
            1 0 0]); % 击不中

g = bwhitmiss(f, B1, B2);
imshowMy(g)
title('使用结构元素组[1]击中击不中变换后的图像')

g2 = bwhitmiss(f, B3, B4);
imshowMy(g2)
title('使用结构元素组[2]击中击不中变换后的图像')

g3 = bwhitmiss(f, B5, B6);
imshowMy(g3)
title('使用结构元素组[3]击中击不中变换后的图像')

%% 利用定义来实现“击中击不中”运算
clc
clear

f = imread('..\Pictures\images_ch09\Fig0913(a)(small-squares).tif');
imshowMy(f)
imfinfoMy(f)
title('原始图像')

B1 = strel([0 0 0; 
            0 1 1; 
            0 1 0]); % 击中
B2 = strel([1 1 1; 
            1 0 0; 
            1 0 0]); % 击不中

B3 = strel([0 1 0; 
            1 1 1; 
            0 1 0]);
B4 = strel([1 0 1; 
            0 0 0; 
            0 0 0]);

g = imerode(f,B1) & imerode(~f,B2);
imshowMy(g)
title('使用结构元素组[1]击中击不中变换后的图像')


%% makelut
clc
clear

f = inline('sum(x(:)) >= 3');
lut2 = makelut(f,2)

lut3 = makelut(f,3)

%% 例9.6 makelut 大笑猫之生存游戏 P267
clc
clear

lut = makelut(@conwaylaws, 3);
bw1 = [0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0
       0     0     0     1     0     0     1     0     0     0
       0     0     0     1     1     1     1     0     0     0
       0     0     1     0     0     0     0     1     0     0
       0     0     1     0     1     1     0     1     0     0
       0     0     1     0     0     0     0     1     0     0
       0     0     0     1     1     1     1     0     0     0
       0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0  ];

imshowMy(bw1, 'notruesize'), title('Generation 1') % 'n' == 'notruesize'

% imshowMy(bw1), title('Generation 1')

bw2 = applylut(bw1, lut);
imshowMy(bw2, 'n'), title('Generation 2')

bw3 = applylut(bw2, lut);
imshowMy(bw3, 'n'), title('Generation 3')

bw4 = applylut(bw3, lut);
imshowMy(bw4, 'n'), title('Generation 4')

bw5 = applylut(bw4, lut);
imshowMy(bw5, 'n'), title('Generation 5')

bw6 = applylut(bw5, lut);
imshowMy(bw6, 'n'), title('Generation 6')

bw7 = applylut(bw6, lut);
imshowMy(bw7, 'n'), title('Generation 7')

bw8 = applylut(bw7, lut);
imshowMy(bw8, 'n'), title('Generation 8')

temp = bw1;
for i = 2:100;
    bw100 = applylut(temp, lut);
    temp = bw100;
end
imshowMy(bw100, 'n'), title('Generation 100')

%% getsequence
clc
clear

se = strel('diamond', 5)
decomp = getsequence(se)
decomp(1)
decomp(2)
decomp(3)
decomp(4)
% decomp(1)*decomp(2)*decomp(3)*decomp(4) 是错误的
% rse = imdilate(imdilate(imdilate(decomp(1), decomp(2)), decomp(3)),...
% decomp(4)) 是错误的

%% endpoints 
clc
clear

f = imread('..\Pictures\images_ch09\Fig0914(a)(bone-skel).tif');
imshowMy(f)
title('原始形态骨骼的图像')

g = endpoints(f);
imshowMy(g)
title('使用函数[endpoints]后得到的端点图像')

f = imread('..\Pictures\images_ch09\Fig0916(a)(bone).tif');
imshowMy(f)
title('原始骨头图像')

g = endpoints(f);
imshowMy(g)
title('使用函数[endpoints]后得到的端点图像（什么也没有）')

%% bwmorph   'remove' 'skel' 'shrink' ‘thin’
clc
clear

BW = imread('circles.png');
imshowMy(BW);

BW2 = bwmorph(BW,'remove');
imshowMy(BW2)

BW3 = bwmorph(BW,'skel',Inf);
imshowMy(BW3)

BW4 = bwmorph(BW,'shrink',Inf); % 去掉毛刺的“骨骼化”
imshowMy(BW4)

ginf = bwmorph(BW, 'thin', Inf); 
imshowMy(ginf)
title('使用函数[bwmorph]细化到稳定状态后的图像')

%% P433 边界提取 bwmorph   'remove' 'skel' 'shrink' ‘thin’

clc
clear

BW = imread('..\DIP_SourceImage\images_chapter_09\Fig9.14(a).jpg');
imshowMy(BW)
title('原始图像')

BW2 = bwmorph(BW,'remove'); % 有时达不到理想的提取边界效果
imshowMy(BW2)

BW3 = bwmorph(BW,'skel',Inf);
imshowMy(BW3)

BW4 = bwmorph(BW,'shrink',Inf); % 去掉毛刺的“骨骼化”
imshowMy(BW4)

ginf = bwmorph(BW, 'thin', Inf); 
imshowMy(ginf)
title('使用函数[bwmorph]细化到稳定状态后的图像')

%% bwmorph   'bridge' 'clean' 'hbreak'
clc
clear

f = imread('..\Pictures\images_ch09\Fig0911(a)(noisy-fingerprint).tif');

imshowMy(f)  % 原始图
title('原始图像')

BW3 = bwmorph(f,'bridge',Inf);
imshowMy(BW3)

BW3 = bwmorph(BW3,'hbreak',Inf); % 极其细微的变化
imshowMy(BW3)

BW3 = bwmorph(f,'clean',Inf);
imshowMy(BW3)


%% bwmorph   'thin' 指纹图像 细化1
clc
clear

f = imread('..\Pictures\images_ch09\Fig0911(a)(noisy-fingerprint).tif');
% f = imread('..\Pictures\Beautiful\hehe1.tif');
imshowMy(f)
title('原始指纹图像')

g1 = bwmorph(f, 'thin', 1);
imshowMy(g1)
title('使用函数[bwmorph]细化一次后的图像')

g2 = bwmorph(f, 'thin', 2);
imshowMy(g2)
title('使用函数[bwmorph]细化两次后的图像')

% 细化到稳定状态
ginf = bwmorph(f, 'thin', Inf); 
imshowMy(ginf)
title('使用函数[bwmorph]细化到稳定状态后的图像')

%% bwmorph   'thin' 骨头图像 细化2
clc
clear

f = imread('..\Pictures\images_ch09\Fig0916(a)(bone).tif');
% f = imread('..\Pictures\Beautiful\hehe1.tif');
imshowMy(f)
title('原始骨头图像')

g1 = bwmorph(f, 'thin', 1);
imshowMy(g1)
title('使用函数[bwmorph]细化一次后的图像')

g2 = bwmorph(f, 'thin', 2);
imshowMy(g2)
title('使用函数[bwmorph]细化两次后的图像')

% 细化到稳定状态
ginf = bwmorph(f, 'thin', Inf); 
imshowMy(ginf)
title('使用函数[bwmorph]细化到稳定状态后的图像')
%% bwmorph  'skel' 骨头图像 骨骼化1
clc
clear

f = imread('..\Pictures\images_ch09\Fig0916(a)(bone).tif');
% f = imread('..\Pictures\Beautiful\hehe1.tif');
imshowMy(f)
title('原始骨头图像')

g1 = bwmorph(f, 'skel', 1);
imshowMy(g1)
title('使用函数[bwmorph]骨骼化一次后的图像')

g2 = bwmorph(f, 'skel', 2);
imshowMy(g2)
title('使用函数[bwmorph]骨骼化两次后的图像')

% 骨骼化到稳定状态
fs = bwmorph(f, 'skel', Inf); 
imshowMy(fs)
title('使用函数[bwmorph]骨骼化到稳定状态后的图像')

for k = 1:5
    fs = fs & ~endpoints(fs);
end
imshowMy(fs)
title('使用函数[endpoints]五次修剪骨骼端点后的图像')

%% bwmorph  'skel' 指纹图像 骨骼化2
clc
clear

f = imread('..\Pictures\images_ch09\Fig0911(a)(noisy-fingerprint).tif');
% f = imread('..\Pictures\Beautiful\hehe1.tif');
imshowMy(f)
title('原始指纹图像')

g1 = bwmorph(f, 'skel', 1);
imshowMy(g1)
title('使用函数[bwmorph]骨骼化一次后的图像')

g2 = bwmorph(f, 'skel', 2);
imshowMy(g2)
title('使用函数[bwmorph]骨骼化两次后的图像')

% 骨骼化到稳定状态
fs = bwmorph(f, 'skel', Inf); 
imshowMy(fs)
title('使用函数[bwmorph]骨骼化到稳定状态后的图像')

for k = 1:5
    fs = fs & ~endpoints(fs);
end
imshowMy(fs)
title('使用函数[endpoints]五次修剪骨骼端点后的图像')
%% 例9.7 bwlabel 标注连通分量
clc
clear
f = imread('..\Pictures\images_ch09\Fig0917(a)(ten-objects).tif');
imshowMy(f)
title('原始图像')

[L4,n4] = bwlabel(f,4);  % 14个4连通分量
imshowMy(L4,[]) % 其中第12 13 14分量在 C 的拐角处！！！
[L,n] = bwlabel(f);      % 10个8连通分量
imshowMy(L,[])


imshowMy(f) % 叠加
hold on     % 叠加
[r,c] = find(L == 3);
rbar = mean(r);
cbar = mean(c);
% plot(cbar,rbar,'Marker','o','MarkerEdgeColor','k',...
%         'MarkerFaceColor','k','MarkerSize',10);
plot(cbar,rbar,'Marker','*','MarkerFaceColor','w');
title('标记第三个对象（连通分量）的质心后的图像')

imshowMy(f) % 叠加
hold on     % 叠加
for k =1:n
    [r,c] = find(L == k);
    rbar = mean(r);
    cbar = mean(c);
%     plot(cbar,rbar,'Marker','o','MarkerEdgeColor','k',...
%         'MarkerFaceColor','k','MarkerSize',10);
    plot(cbar,rbar,'Marker','*','MarkerFaceColor','w');
end
title('使用循环标记所有对象（连通分量）的质心后的图像')



%% 例9.8 imreconstruct imfill imclearborder
clc
clear
f = imread('..\Pictures\images_ch09\Fig0922(a)(book-text).tif');
imshowMy(f)
title('原始图像')

fe = imerode(f,ones(51,1));
imshowMy(fe)
title('使用竖线腐蚀后的图像')

fo = imopen(f,ones(51,1));
imshowMy(fo)
title('使用竖线做开运算后的图像')

fobr = imreconstruct(fe,f);
imshowMy(fobr)
title('使用竖线做重构后的图像')

g = imfill(f,'holes');
imshowMy(g)
title('填充孔洞（完全封闭）后的图像')

g1 = imclearborder(f,8);
imshowMy(g1)
title('清楚边界对象后的图像')



%% 例9.9 使用开运算和闭运算进行形态学平滑
clc
clear
f = imread('..\Pictures\images_ch09\Fig0925(a)(dowels).tif');
imshowMy(f)
title('木按钉的原始图像')

se = strel('disk',5);
fo = imopen(f,se);
imshowMy(fo)
title('使用半径为5的圆盘执行开运算后的图像')

foc = imclose(fo,se);
imshowMy(foc)
title('经过开运算后再经闭运算后的图像')

fasf = f;
for k = 2:5
    se = strel('disk',k);
    fasf = imclose(imopen(fasf,se),se);
end
imshowMy(fasf)
title('交替顺序滤波后的图像')

%% 例9.10 使用顶帽变换 和 底帽变换
clc
clear
f = imread('..\Pictures\images_ch09\Fig0926(a)(rice).tif');
imshowMy(f)
title('原始图像')

T1 = 255*graythresh(f)
g = f>=T1;
imshowMy(g)
title('经过阈值处理后的图像')

se = strel('disk',10);
fo = imopen(f,se);
imshowMy(fo)
title('经过开运算后的图像')

f2 = imsubtract(f,fo);
imshowMy(f2)
title('顶帽变换(原始图像减去经过开运算后的图像)')

T2 = 255*graythresh(f2) % 自动获得阈值
g1 = f2>=T2;
imshowMy(g1)
title('经过阈值处理后的顶帽变换图像')

f2 = imtophat(f,se); % 使用顶帽变换
imshowMy(f2)
title('顶帽变换[使用 imtophat 函数]')

se1 = strel('disk',10);
f3 = imbothat(imcomplement(f),se1); % 使用底帽变换
imshowMy(f3)
title('底帽变换[使用 imbothat 函数]')

se = strel('disk',3);
g2 = imsubtract(imadd(f,imtophat(f,se)),imbothat(f,se));
imshowMy(g2)
title('顶帽变换和底帽变换联合使用（用于增强对比度）')

%% 顶帽变换一个例子 imtophat + imadjust
clc
clear

I = imread('rice.png');
imshowMy(I)

K1 = imadjust(I);
imshowMy(K1)

se = strel('disk',12);
J = imtophat(I,se);
imshowMy(J)

K2 = imadjust(J);
imshowMy(K2)

%% 顶帽变换和底帽变换联合使用（用于增强对比度）
clc
clear
I = imread('pout.tif');
imshowMy(I)

se = strel('disk',3);

J = imsubtract(imadd(I,imtophat(I,se)),imbothat(I,se));
imshowMy(J)
title('顶帽变换和底帽变换联合使用（用于增强对比度）')


%% 例9.11 颗粒分析 米粒图像
clc
clear
f = imread('..\Pictures\images_ch09\Fig0926(a)(rice).tif');
imshowMy(f)
title('原始图像')

sumpixels = zeros(1,36);
for k = 0:35
    se = strel('disk',k);
    fo = imopen(f,se);
    sumpixels(k + 1) = sum(fo(:));
end
figure,plot(0:35,sumpixels)
xlabel('k'),ylabel('Surface area')
set(gcf,'outerposition',get(0,'screensize'))

figure,plot(-diff(sumpixels)) % 表示半径尺寸为5的对象最多，其次为尺寸6
xlabel('k'),ylabel('Surface area reduction')
set(gcf,'outerposition',get(0,'screensize'))

%% 例9.11 颗粒分析 图钉图像（时间较长）
clc
clear
f = imread('..\Pictures\images_ch09\Fig0925(a)(dowels).tif');
imshowMy(f)
title('木按钉的原始图像')

se = strel('disk',5);
fo = imopen(f,se);
imshowMy(fo)
title('使用半径为5的圆盘执行开运算后的图像')

foc = imclose(fo,se);
imshowMy(foc)
title('经过开运算后再经闭运算后的图像')

fasf = f;
for k = 2:5
    se = strel('disk',k);
    fasf = imclose(imopen(fasf,se),se);
end
imshowMy(fasf)
title('交替顺序滤波后的图像')

sumpixels = zeros(1,36);
for k = 0:35
    se = strel('disk',k);
    fo = imopen(f,se);
    sumpixels(k + 1) = sum(fo(:));
end
figure,plot(0:35,sumpixels)
xlabel('k'),ylabel('Surface area')
set(gcf,'outerposition',get(0,'screensize'))

figure,plot(-diff(sumpixels)) % 表示半径尺寸为5的对象最多，其次为尺寸6
xlabel('k'),ylabel('Surface area reduction')
set(gcf,'outerposition',get(0,'screensize'))

f = fasf;
sumpixels = zeros(1,36);
for k = 0:35
    se = strel('disk',k);
    fo = imopen(f,se);
    sumpixels(k + 1) = sum(fo(:));
end
figure,plot(0:35,sumpixels)
xlabel('k'),ylabel('Surface area')
set(gcf,'outerposition',get(0,'screensize'))

figure,plot(-diff(sumpixels)) % 表示半径尺寸为5的对象最多，其次为尺寸6
xlabel('k'),ylabel('Surface area reduction')
set(gcf,'outerposition',get(0,'screensize'))

%% P283 灰度重构(删除比结构元素小的对象)
clc
clear
f = imread('..\Pictures\images_ch09\Fig0925(a)(dowels).tif');
imshowMy(f)
title('原始图像')

se = strel('disk',5);
fe = imerode(f,se); % 用作标记图像
fobr = imreconstruct(fe,f);
imshowMy(fobr)
title('开运算重构后的图像')

fobrc = imcomplement(fobr);
fobrce = imerode(fobrc,se);
fobrcbr = imcomplement(imreconstruct(fobrce,fobrc));
imshowMy(fobrcbr)
title('经开运算重构后再经闭运算重构的图像')

fobrc = imcomplement(f);
fobrce = imerode(fobrc,se);
fobrcbr = imcomplement(imreconstruct(fobrce,fobrc));
imshowMy(fobrcbr)
title('直接进行闭运算重构的图像')

%% 例9.12 使用重构删除复杂图像的背景
clc
clear
f = imread('..\Pictures\images_ch09\Fig0930(a)(calculator).tif');
imshowMy(f)
title('原始图像')

f_obr = imreconstruct(imerode(f,ones(1,71)),f);
imshowMy(f_obr)
title('经过腐蚀运算重构后的图像（结构元素：使用一条长水平线）')

% f_obr = imreconstruct(imopen(f,ones(1,71)),f);
% imshowMy(f_obr)
% title('经过开运算重构后的图像（结构元素：使用一条长水平线）')

f_o = imopen(f,ones(1,71));
imshowMy(f_o)
title('经过标准开运算后的图像（结构元素：使用一条长水平线）')

f_thr = imsubtract(f,f_obr);
imshowMy(f_thr)
title('经过开运算重构后的图像（掩模图像）')

f_th = imsubtract(f,f_o);
imshowMy(f_th)
title('经过顶帽变换后的图像')

g_obr = imreconstruct(imerode(f_thr,ones(1,11)),f_thr);
imshowMy(g_obr)
title('对[经过开运算重构后的图像]使用一条水平线开运算重构后的图像')

g_obrd = imdilate(g_obr,ones(1,21));
imshowMy(g_obrd)
title('使用一条水平线对上一张图像进行膨胀后的图像')

temp = min(g_obrd,f_thr);
imshowMy(temp)
title('标记图像')

f2 = imreconstruct(min(g_obrd,f_thr),f_thr);
imshowMy(f2)
title('最后的重构结果')

%% 例9.12 使用重构删除复杂图像的背景（连贯步骤）
clc
clear
f = imread('..\Pictures\images_ch09\Fig0930(a)(calculator).tif');
imshowMy(f)
title('原始图像')

f_obr = imreconstruct(imerode(f,ones(1,71)),f);
imshowMy(f_obr)
title('经过腐蚀运算重构后的图像（结构元素：使用一条长水平线）')

f_thr = imsubtract(f,f_obr);
imshowMy(f_thr)
title('经过开运算重构后的图像（掩模图像）')

g_obr = imreconstruct(imerode(f_thr,ones(1,11)),f_thr);
imshowMy(g_obr)
title('对[经过开运算重构后的图像]使用一条水平线开运算重构后的图像')

g_obrd = imdilate(g_obr,ones(1,21));
imshowMy(g_obrd)
title('使用一条水平线对上一张图像进行膨胀后的图像')

temp = min(g_obrd,f_thr);
imshowMy(temp)
title('标记图像')

f2 = imreconstruct(min(g_obrd,f_thr),f_thr);
imshowMy(f2)
title('最后的重构结果')


%% 编程笔记附录

%% 击中击不中两个公式等价编程实现
击中击不中公式 1： imerode(bw,se1) & ~imdilate(bw,reflect(se2))
击中击不中公式 2： imerode(bw,se1) & imerode(~bw,se2)
%% 形态学函数大总结

1、 首先 imerode 和 imdilate 是最重要的两个形态学函数，
    因为几乎所有的其他函数都可以通过简单的 &, |, ~, reflect, -, imerode，imdilate来实现。
    其次 imerode，imdilate 的具体实现又是通过函数 morphop 来实现的。

    在前人的基础上，我们的工作只需要理解各种操作的公式和物理意义！

    但是这些理论只是纸上谈兵，千万不能忽视编程基本功！

2、 imreconstruct imfill imclearborder 三个函数，imfill imclearborder 都使用了 imreconstruct，而 imreconstruct 是通过 C++ 实现的。

3、 形态学重构的作用：只留下包括结构元素的对象。

4、 灰度图像开运算的作用：可以除去比结构元素更小的明亮细节，同时保持图像整体的灰度级和较大的明亮区域不变。
    灰度图像闭运算的作用：可以除去比结构元素更小的暗部细节，同时保持图像整体的灰度级和较大的暗部区域不变。
   
5、 形态学平滑：先开运算再闭运算，可以同时除去人为的亮和暗的因素或噪声。

6、 顶帽变换：对于增强阴影的细节很有用处，还有对亮度不均匀图像作暗度补偿。
    顶帽变换和底帽变换联合使用（用于增强对比度）
   
7、 灰度级形态学重构：可以用于消除比结构元素小的对象，而且很好的维护了原始图像概貌。   
   
8、 连通性在解决分类问题上发挥着着很重要的作用！！！   

9、 使用同样的算子，一幅已经进行过一次“开操作”的图像紧接着再进行多少次“开操作”也不会有变化了。对于“闭操作”同理之。
    对于同一幅图像“先开再闭”和“先闭再开”结果“并不是相同的”，由此可知，“开操作”和“闭操作”“不”具有“顺序”交换性
   
   
   
   
   
   
   
   