%% 第十章 图像分割

%% 例10.1 点检测
clc
clear
f = imread('..\Pictures\images_ch10\Fig1002(a)(test_pattern_with_single_pixel).tif');
imshowMy(f)
title('原始图像')

w = [-1 -1 -1;
     -1  8 -1; 
     -1 -1 -1];
g = abs(imfilter(double(f),w));
% **** 求矩阵最大值和对应坐标位置 ********
[c1,i] = max(g);
[c2,j] = max(c1);
max_ans = g(i(j),j)      % 最大值
max_location = [i(j),j]  % 最大值坐标位置
% **************************************
T = max(g(:));
g1 = g>=T/100;
g2 = g>=T;
% **** 求矩阵最大值和对应坐标位置 ********
[c1,i] = max(g);
[c2,j] = max(c1);
max_ans = g(i(j),j)      % 最大值
max_location = [i(j),j]  % 最大值坐标位置
% **************************************
% figure,imshow(g)
imshowMy(g1)
title('采用界限值[g>=T/100]后的图像')

imshowMy(g2)
title('采用界限值[g>=T]后的图像')

%% 例10.1 点检测结果有出入 ordfilt2 一个点变成三个点
clc
clear
f = imread('..\Pictures\images_ch10\Fig1002(a)(test_pattern_with_single_pixel).tif');
imshowMy(f)
title('原始图像')

m=3;
n=3;

g = imsubtract(ordfilt2(f, m*n, ones(m, n)), ordfilt2(f, 1, ones(m, n)));
T = max(g(:));
g2 = g>=T;
imshowMy(g2)

%% 例10.2 线检测
clc
clear
f = imread('..\Pictures\images_ch10\Fig1004(a)(wirebond_mask).tif');
imshowMy(f)
title('原始连线掩模图像')

w = [2 -1 -1;
     -1 2 -1; 
     -1 -1 2];
g = imfilter(double(f),w);
imshowMy(g,[])
title('使用[-45度（请参见 P288）]检测器处理后的图像')

gtop = g(1:120,1:120);
gtop = pixeldup(gtop,4);
imshowMy(gtop,[])
title('{使用[-45度（请参见 P288）]检测器处理后的图像}左上角的放大图')

gbot = g(end-119:end,end-119:end);
gbot = pixeldup(gbot,4);
imshowMy(gbot,[])
title('{使用[-45度（请参见 P288）]检测器处理后的图像}右下角的放大图')

g = abs(g);
imshowMy(g,[])
title('{使用[-45度（请参见 P288）]检测器处理后的图像}的绝对值')

T = max(g(:));
g = g>=T;
imshowMy(g)
title('满足条件[g>=T]的所有点（白色点）[其中g是上一张图片]')

%% 例10.3 edge 边缘检测 sobel
clc
clear
f = imread('..\Pictures\images_ch10\Fig1006(a)(building).tif');
imshowMy(f)
title('原始图像')

[gv1,tvertical] = edge(f,'sobel','vertical');
imshowMy(gv1)
tvertical
title('使用带有自动确定的阈值的一个垂直[sobel]掩模后，函数[edge]导致的结果')

[gv1,thorizontal] = edge(f,'sobel','horizontal');
imshowMy(gv1)
thorizontal
title('使用带有自动确定的阈值的一个水平[sobel]掩模后，函数[edge]导致的结果')

gv2 = edge(f,'sobel',0.15,'vertical');
imshowMy(gv2)
title('使用指定阈值的一个垂直[sobel]掩模后，函数[edge]导致的结果')

gboth = edge(f,'sobel',0.15);
imshowMy(gboth)
title('使用指定阈值的一个同时考虑水平垂直[sobel]掩模后，函数[edge]导致的结果')

[gboth,tboth] = edge(f,'sobel');
tboth
imshowMy(gboth)
title('使用自动阈值的一个同时考虑水平垂直[sobel]掩模后，函数[edge]导致的结果')

w45 = [-2 -1 0; -1 0 1; 0 1 2];
g45 = imfilter(double(f),w45,'replicate');
T = 0.3*max(abs(g45(:)));
g45 = g45>=T;
imshowMy(g45)
title('使用函数[imfilter]（具有指定的掩模和阈值）计算[45度]边缘的结果')

w_45 = [0 1 2; -1 0 1; -2 -1 0];
g_45 = imfilter(double(f),w_45,'replicate');
T = 0.3*max(abs(g_45(:)));
g_45 = g_45>=T;
imshowMy(g_45)
title('使用函数[imfilter]（具有指定的掩模和阈值）计算[-45度]边缘的结果')

imshowMy(g45 + g_45)
title('使用函数[imfilter]（具有指定的掩模和阈值）计算[-45度]+[45度]边缘的结果')

%% 例10.3 edge 边缘检测 prewitt
clc
clear
f = imread('..\Pictures\images_ch10\Fig1006(a)(building).tif');
imshowMy(f)
title('原始图像')

[gv1,t] = edge(f,'prewitt','vertical');
imshowMy(gv1)
t
title('使用带有自动确定的阈值的一个垂直[prewitt]掩模后，函数[edge]导致的结果')

gv2 = edge(f,'prewitt',0.15,'vertical');
imshowMy(gv2)
title('使用指定阈值的一个垂直[prewitt]掩模后，函数[edge]导致的结果')

gboth = edge(f,'prewitt',0.15);
imshowMy(gboth)
title('使用指定阈值的一个同时考虑水平垂直[prewitt]掩模后，函数[edge]导致的结果')

w45 = [-2 -1 0; -1 0 1; 0 1 2]
g45 = imfilter(double(f),w45,'replicate');
T = 0.3*max(abs(g45(:)));
g45 = g45>=T;
imshowMy(g45)
title('使用函数[imfilter]（具有指定的掩模和阈值）计算[45度]边缘的结果')

w_45 = [0 1 2; -1 0 1; -2 -1 0]
g_45 = imfilter(double(f),w_45,'replicate');
T = 0.3*max(abs(g_45(:)));
g_45 = g_45>=T;
imshowMy(g_45)
title('使用函数[imfilter]（具有指定的掩模和阈值）计算[-45度]边缘的结果')

%% 例10.3 edge 边缘检测 roberts
clc
clear
f = imread('..\Pictures\images_ch10\Fig1006(a)(building).tif');
imshowMy(f)
title('原始图像')

[gv1,t] = edge(f,'roberts','vertical');
imshowMy(gv1)
t
title('使用带有自动确定的阈值的一个垂直[roberts]掩模后，函数[edge]导致的结果')

gv2 = edge(f,'roberts',0.15,'vertical');
imshowMy(gv2)
title('使用指定阈值的一个垂直[roberts]掩模后，函数[edge]导致的结果')

gboth = edge(f,'roberts',0.15);
imshowMy(gboth)
title('使用指定阈值的一个同时考虑水平垂直[roberts]掩模后，函数[edge]导致的结果')

w45 = [-2 -1 0; -1 0 1; 0 1 2]
g45 = imfilter(double(f),w45,'replicate');
T = 0.3*max(abs(g45(:)));
g45 = g45>=T;
imshowMy(g45)
title('使用函数[imfilter]（具有指定的掩模和阈值）计算[45度]边缘的结果')

w_45 = [0 1 2; -1 0 1; -2 -1 0]
g_45 = imfilter(double(f),w_45,'replicate');
T = 0.3*max(abs(g_45(:)));
g_45 = g_45>=T;
imshowMy(g_45)
title('使用函数[imfilter]（具有指定的掩模和阈值）计算[-45度]边缘的结果')

%% 例10.4 Sobel, LoG, Canny 边缘检测器的比较
clc
clear
f = imread('..\Pictures\images_ch10\Fig1006(a)(building).tif');
imshowMy(f)
title('原始图像')

[g_sobel_default,ts] = edge(f,'sobel');
imshowMy(g_sobel_default)
title('[sobel]边缘检测器使用默认选项产生的结果')

[g_log_default,tlog] = edge(f,'log');
imshowMy(g_log_default)
title('[LoG]边缘检测器使用默认选项产生的结果')

[g_canny_default,tc] = edge(f,'canny');
imshowMy(g_canny_default)
title('[canny]边缘检测器使用默认选项产生的结果')

[g_sobel_best,ts] = edge(f,'sobel',0.05);
imshowMy(g_sobel_best)
title('[sobel]边缘检测器使用 edge(f,sobel,0.05) 产生的结果')

[g_log_best,tlog] = edge(f,'log',0.003,2.25);
imshowMy(g_log_best)
title('[LoG]边缘检测器使用 edge(f,log,0.003,2.25) 产生的结果')

[g_canny_best,tc] = edge(f,'canny',[0.04 0.10],1.5);
imshowMy(g_canny_best)
title('[canny]边缘检测器使用 edge(f,canny,[0.04 0.10],1.5) 产生的结果')

%% 例10.5 Hough 变换
clc
clear
f = zeros(101,101);
f(1,1) = 1;
f(101,1) = 1;
f(1,101) = 1;
f(51,51) = 1;
f(101,101) = 1;
imshowMy(f)
title('带有五个点的二值图像')

H = hough(f);
imshowMy(H,[])
title('Hough 变换')

[H,theta,rho] = hough(f);
imshowMy(theta,rho,H,[],'notruesize')
axis on, axis normal
xlabel('\theta'),ylabel('\rho')
title('带有标度轴的 Hough 变换')

%% 例10.6 Hough变换 线检测和链接 hough houghpeaks houghlines
clc
clear
f1 = imread('..\Pictures\images_ch10\Fig1006(a)(building).tif');
imshowMy(f1)
title('原始图像')

[f,tc] = edge(f1,'canny',[0.04 0.10],1.5);
imshowMy(f)
title('[canny]边缘检测器使用 edge(f,canny,[0.04 0.10],1.5) 产生的结果')

[H,theta,rho] = hough(f,0.5); %
imshowMy(theta,rho,H,[],'notruesize')
axis on, axis normal
xlabel('\theta'),ylabel('\rho')


[r,c] = houghpeaks(H,10);
hold on
plot(theta(c),rho(r),'linestyle','none',...
    'marker','s','color','w')
title('带有所选10个峰值的位置的 Hough 变换')

lines = houghlines(f,theta,rho,r,c)
imshowMy(f),hold on
for k = 1:length(lines)
    xy = [lines(k).point1;lines(k).point2];
    plot(xy(:,2),xy(:,1),'LineWidth',4,'Color',[0.6 0.6 0.6]);
end
title('Hough 变换峰值对应的线段')

[H,theta,rho] = hough(f,0.5); %
imshowMy(theta,rho,H,[],'notruesize')
axis on, axis normal
xlabel('\theta'),ylabel('\rho')

[r,c] = houghpeaks(H,100,1);
hold on
plot(theta(c),rho(r),'linestyle','none',...
    'marker','s','color','w')
title('带有所选100个峰值的位置的 Hough 变换')

lines = houghlines(f,theta,rho,r,c)
imshowMy(f),hold on
for k = 1:length(lines)
    xy = [lines(k).point1;lines(k).point2];
    plot(xy(:,2),xy(:,1),'LineWidth',4,'Color',[0.6 0.6 0.6]);
end
title('Hough 变换峰值对应的线段')

lines = houghlines(f,theta,rho,r,c)
imshowMy(zeros(size(f))),hold on
for k = 1:length(lines)
    xy = [lines(k).point1;lines(k).point2];
    plot(xy(:,2),xy(:,1),'LineWidth',4,'Color',[0.6 0.6 0.6]);
end
title('Hough 变换峰值对应的线段')

%% 例10.7 计算全局阈值 graythresh
clc
clear
f = imread('..\Pictures\images_ch10\Fig1013(a)(scanned-text-grayscale).tif');
imshowMy(f)
title('原始图像')
imhistMy(f)

T = 0.5*(double(min(f(:))) + double(max(f(:))));
done = false;
while ~done
    g = f>=T;
    Tnext = 0.5*(mean(f(g)) + mean(f(~g)));
    done = abs(T - Tnext) < 0.5;
    T = Tnext;
end
T
g = f<=T;
imshowMy(g)
title('使用迭代方法得到的阈值处理后的图像')

T2 = graythresh(f)
g = f<=T2*255;
imshowMy(g)
title('使用函数[graythresh]得到的阈值处理后的图像')

T2*255

%% bwlabel 在一个二值图像中标注连通分量
clc
clear

BW = [1     1     1     0     0     0     0     0
      1     1     1     0     1     1     0     0
      1     1     1     0     1     1     0     0
      1     1     1     0     0     0     1     0
      1     1     1     0     0     0     1     0
      1     1     1     0     0     0     1     0
      1     1     1     0     0     1     1     0
      1     1     1     0     0     0     0     0];

[L8,n8] = bwlabel(BW,8) % 默认为 8 连通的: bwlabel(BW,8) == bwlabel(BW)

[L4,n4] = bwlabel(BW,4)

%% 例10.8 区域生长 regiongrow 房子
clc
clear
f = imread('..\Pictures\images_ch10\Fig1006(a)(building).tif');
imshowMy(f)
title('原始图像')

[g,NR,SI,TI] = regiongrow(f,100,10);

NR
imshowMy(SI) % 包含有种子点的图像
title('种子点图像')

imshowMy(TI) % 包含在经过连通性处理前通过阈值测试的像素
title('所有[经过种子点连通性处理前]通过阈值测试的像素')

imshowMy(g)
title('所有通过阈值测试的像素在对种子点进行8连通性分析后的结果')

%% 例10.8 区域生长 regiongrow 焊接空隙
clc
clear
f = imread('..\Pictures\images_ch10\Fig1014(a)(defective_weld).tif');
imshowMy(f)
title('原始图像')

[g,NR,SI,TI] = regiongrow(f,255,65);

NR
imshowMy(SI) % 包含有种子点的图像
title('种子点图像')

imshowMy(TI) % 包含在经过连通性处理前通过阈值测试的像素
title('所有[经过种子点连通性处理前]通过阈值测试的像素')

imshowMy(g)
title('所有通过阈值测试的像素在对种子点进行8连通性分析后的结果')

%% qtdecomp 执行四茶树分解
clc
clear
I = imread('liftingbody.png');
S = qtdecomp(I,0.99);
blocks = repmat(uint8(0),size(S));

for dim = [512 256 128 64 32 16 8 4 2 1];    
  numblocks = length(find(S==dim));    
  if (numblocks > 0)        
    values = repmat(uint8(1),[dim dim numblocks]);
    values(2:dim,2:dim,:) = 0;
    blocks = qtsetblk(blocks,S,dim,values);
  end
end

blocks(end,1:end) = 1;
blocks(1:end,end) = 1;

imshowMy(I)
title('原始图像')

imshowMy(blocks,[])

imshowMy(I.*(1-blocks))

%% qtdecomp 执行四茶树分解 边界分割
clc
clear
I = imread('liftingbody.png');
S = qtdecomp(I,0.09); % 0.09
blocks = repmat(uint8(0),size(S));

for dim = [1];    % 只标出为 1*1
  numblocks = length(find(S==dim));    
  if (numblocks > 0)        
    values = repmat(uint8(1),[dim dim numblocks]);
    values(2:dim,2:dim,:) = 0;
    blocks = qtsetblk(blocks,S,dim,values);
  end
end

blocks(end,1:end) = 1;
blocks(1:end,end) = 1;

imshowMy(I)
title('原始图像')

imshowMy(blocks,[])

imshowMy(I.*(1-blocks))

%% qtdecomp 执行四茶树分解 
clc
clear
I = imread('liftingbody.png');
S = qtdecomp(I,0.19); % 参数为 0.19
blocks = repmat(uint8(0),size(S));

for dim = [512 256 128 64 32 16 8 4 2 1];    
  numblocks = length(find(S==dim));    
  if (numblocks > 0)        
    values = repmat(uint8(1),[dim dim numblocks]);
    values(2:dim,2:dim,:) = 0;
    blocks = qtsetblk(blocks,S,dim,values);
  end
end

blocks(end,1:end) = 1;
blocks(1:end,end) = 1;

imshowMy(I)
title('原始图像')

imshowMy(blocks,[])

imshowMy(I.*(1-blocks))

%% qtdecomp qtgetblk
clc
clear
I = [1     1     1     1     2     3     6     6
     1     1     2     1     4     5     6     8
     1     1     1     1    10    15     7     7
     1     1     1     1    20    25     7     7
    20    22    20    22     1     2     3     4
    20    22    22    20     5     6     7     8
    20    22    20    20     9    10    11    12
    22    22    20    20    13    14    15    16]

S = qtdecomp(I,5)
full(S)

[vals,r,c] = qtgetblk(I,S,4) 
[vals,r,c] = qtgetblk(I,S,2) 

newvals = cat(3,zeros(4),ones(4)); 
J = qtsetblk(I,S,4,newvals)

newvals = cat(3,zeros(2),ones(2),ones(2)*2,ones(2)*3,ones(2)*4,ones(2)*5,ones(2)*6); 
J = qtsetblk(I,S,2,newvals)

%% 例10.9 使用区域分离和合并的图像分割 splitmerge qtdecomp qtgetblk
clc
clear
f = imread('..\Pictures\images_ch10\Fig1017(a)(cygnusloop_Xray_original).tif');
imshowMy(f)
title('原始图像')

g = splitmerge(f,32,@predicate);
imshowMy(g)
title('使用函数[splitmerge]且mindim的值为32时进行分割后的图像')

g = splitmerge(f,16,@predicate);
imshowMy(g)
title('使用函数[splitmerge]且mindim的值为16时进行分割后的图像')

g = splitmerge(f,8,@predicate);
imshowMy(g)
title('使用函数[splitmerge]且mindim的值为8时进行分割后的图像')

imshowMy(uint8(double(f).*(1-g))) % 以 8 为掩模时的分割结果

g = splitmerge(f,4,@predicate);
imshowMy(g,[])
max(g(:))
class(g)
title('使用函数[splitmerge]且mindim的值为4时进行分割后的图像')

% double类型时，凡是大于1的数值统统影射为1
g = splitmerge(f,2,@predicate);
imshowMy(g)
max(g(:))
class(g)
title('使用函数[splitmerge]且mindim的值为2时进行分割后的（二值）图像')

% double类型时，凡是大于1的数值统统影射为1
g = splitmerge(f,2,@predicate);
imshowMy(g,[])    % 不是很理解？？？
title('使用函数[splitmerge]且mindim的值为2时进行分割后的（灰度）图像')

%% 例10.9 使用区域分离和合并的图像分割 splitmerge qtdecomp qtgetblk 循环方式 
clc
clear
f = imread('..\Pictures\images_ch10\Fig1017(a)(cygnusloop_Xray_original).tif');
imshowMy(f)
title('原始图像')

for i = 5:-1:1
    dd = 2^i
    g = splitmerge(f,dd,@predicate);
    imshowMy(g)
    name = sprintf('使用函数[splitmerge]且mindim的值为%2d时进行分割后 的图像',dd);
    title(name)
end

%% 大综合1： 开操作 四茶树区域分离与合并 Hough变换

clc
clear
I = imread('liftingbody.png');
S = qtdecomp(I,0.03); % 0.09
blocks = repmat(uint8(0),size(S));

for dim = [1];    % 只标出为 1*1
  numblocks = length(find(S==dim));    
  if (numblocks > 0)        
    values = repmat(uint8(1),[dim dim numblocks]);
    values(2:dim,2:dim,:) = 0;
    blocks = qtsetblk(blocks,S,dim,values);
  end
end

blocks(end,1:end) = 1;
blocks(1:end,end) = 1;

imshowMy(I)
title('原始图像')

imshowMy(blocks,[])

imshowMy(I.*(1-blocks))

f = blocks>0.5;
se = strel('square', 3);  % 结构元素
f = imopen(f, se);
imshowMy(f)    % 开
title('使用结构元素[square（3）]开操作后的图像')


[H,theta,rho] = hough(f,0.5); %
imshowMy(theta,rho,H,[],'notruesize')
axis on, axis normal
xlabel('\theta'),ylabel('\rho')


[r,c] = houghpeaks(H,10,1);
hold on
plot(theta(c),rho(r),'linestyle','none',...
    'marker','s','color','w')
title('带有所选10个峰值的位置的 Hough 变换')
hold off
lines = houghlines(f,theta,rho,r,c)
imshowMy(f),hold on
for k = 1:length(lines)
    xy = [lines(k).point1;lines(k).point2];
    plot(xy(:,2),xy(:,1),'LineWidth',4,'Color',[0.6 0.6 0.6]);
end
title('Hough 变换峰值对应的线段')

[H,theta,rho] = hough(f,0.5); %
imshowMy(theta,rho,H,[],'notruesize')
axis on, axis normal
xlabel('\theta'),ylabel('\rho')

[r,c] = houghpeaks(H,100,1);
hold on
plot(theta(c),rho(r),'linestyle','none',...
    'marker','s','color','w')
title('带有所选100个峰值的位置的 Hough 变换')

lines = houghlines(f,theta,rho,r,c)
imshowMy(f),hold on
for k = 1:length(lines)
    xy = [lines(k).point1;lines(k).point2];
    plot(xy(:,2),xy(:,1),'LineWidth',4,'Color',[0.6 0.6 0.6]);
end
title('Hough 变换峰值对应的线段')

lines = houghlines(f,theta,rho,r,c)
imshowMy(zeros(size(f))),hold on
for k = 1:length(lines)
    xy = [lines(k).point1;lines(k).point2];
    plot(xy(:,2),xy(:,1),'LineWidth',4,'Color',[0.6 0.6 0.6]);
end
title('Hough 变换峰值对应的线段')

%% 例10.10 用距离和分水岭变换分割灰度图像(有很多小黑点)噪声影响太大了
clc
clear
f = imread('..\Pictures\images_ch10\Fig0925(a)(dowels).tif');
imshowMy(f)
title('原始图像')

% f = double(f);
g = im2bw(f,graythresh(f));
imshowMy(g)

gc = ~g;
imshowMy(gc)

D = bwdist(gc);
imshowMy(D)

L = watershed(-D);
w = L == 0;
imshowMy(w)

g2 = g & ~w;
imshowMy(g2)

%% 大综合2：例10.10 用距离和分水岭变换分割灰度图像(有很多小黑点)噪声影响减小了（预先进行开操作处理和填充孔洞、闭操作）
clc
clear
f = imread('..\Pictures\images_ch10\Fig0925(a)(dowels).tif');
imshowMy(f)
title('原始图像')

% f = double(f);
g = im2bw(f,graythresh(f));
imshowMy(g)

se = strel('square', 3);  % 结构元素
g = imopen(g, se);
imshowMy(g)    % 开
title('使用结构元素[square（3）]开操作后的图像')

g = imfill(im2bw(g),'holes');
imshowMy(g)
title('填充孔洞（完全封闭）后的图像')

g = imclose(g, se);
imshowMy(g)    % 开
title('使用结构元素[square（3）]开操作后的图像')

gc = ~g;
imshowMy(gc)

D = bwdist(gc);
imshowMy(D)

L = watershed(-D);
w = L == 0;
imshowMy(w)

g2 = g & ~w;
imshowMy(g2)

g3 = xor(g,w);
imshowMy(g3)
%% 例10.10 用距离和分水岭变换分割二值图像
clc
clear
g = imread('..\Pictures\images_ch10\Fig1020(a)(binary-dowel-image).tif');
imshowMy(g)
title('原始图像')

% % f = double(f);
% g = im2bw(f,graythresh(f));
% imshowMy(g)

gc = ~g;
imshowMy(gc)
title('原始图像的补')

D = bwdist(gc);
imshowMy(D,[])
title('距离变换')

imshowMy(-D,[])
title('距离变换的补')

L = watershed(-D);
imshowMy(L,[])
title('距离变换的负分水岭脊线（灰度图）')

max(L(:))
min(L(:))
class(L)

w = L == 0;
imshowMy(w)
title('距离变换的负分水岭脊线（二值图）')

g1 = g + w;
imshowMy(g1)
title('白色线叠加在原始二值图像上后的分水岭脊线')

g2 = g & ~w;
imshowMy(g2)
title('黑色线叠加在原始二值图像上后的分水岭脊线（用距离和分水岭变换分割二值图像）')

g3 = xor(g,w);
imshowMy(g3)

%% 例10.11 用梯度和分水岭变换分割灰度图像
clc
clear
f = imread('..\Pictures\images_ch10\Fig1021(a)(small-blobs).tif');
imshowMy(f)
title('原始图像')

h = fspecial('sobel');
fd = double(f);
g = sqrt(imfilter(fd,h,'replicate').^2 + imfilter(fd,h','replicate').^2);
imshowMy(g,[])
max(g(:))
class(g)
title('梯度幅度图像')

L = watershed(g);
imshowMy(L,[])
title('')
title('对梯度幅度图像进行分水岭变换（过分割）（灰度）')

wr = L == 0;
imshowMy(wr,[])
title('对梯度幅度图像进行分水岭变换（过分割）（二值）')

g2 = imclose(imopen(g,ones(3,3)),ones(3,3));
imshowMy(g2,[])
title('对梯度幅度图像用结构元素 ones(3,3) 进行先开再闭操作（平滑处理）')

L2 = watershed(g2);
imshowMy(L2,[])
title('对平滑后的梯度幅度图像进行分水岭变换（少许过分割）（灰度）')

wr2 = L2 == 0;
imshowMy(wr2,[])
title('对平滑后的梯度幅度图像进行分水岭变换（少许过分割）（二值）')

f2 = f;
f2(L2 == 0) = 255;
imshowMy(f2,[])
title('分割线叠加在原始图像上的结果（用梯度和分水岭变换分割灰度图像）')
%% 例10.11 用梯度和分水岭变换分割灰度图像 图钉（梯度图像特别清晰）
clc
clear
f = imread('..\Pictures\images_ch10\Fig0925(a)(dowels).tif');
imshowMy(f)
title('原始图像')

h = fspecial('sobel');
fd = double(f);
g = sqrt(imfilter(fd,h,'replicate').^2 + imfilter(fd,h','replicate').^2);
imshowMy(g,[])
max(g(:))
class(g)
title('梯度幅度图像')

L = watershed(g);
imshowMy(L,[])
title('')
title('对梯度幅度图像进行分水岭变换（过分割）（灰度）')

wr = L == 0;
imshowMy(wr,[])
title('对梯度幅度图像进行分水岭变换（过分割）（二值）')

g2 = imclose(imopen(g,ones(3,3)),ones(3,3));
imshowMy(g2,[])
title('对梯度幅度图像用结构元素 ones(3,3) 进行先开再闭操作（平滑处理）')

L2 = watershed(g2);
imshowMy(L2,[])
title('对平滑后的梯度幅度图像进行分水岭变换（少许过分割）（灰度）')

wr2 = L2 == 0;
imshowMy(wr2,[])
title('对平滑后的梯度幅度图像进行分水岭变换（少许过分割）（二值）')

f2 = f;
f2(L2 == 0) = 255;
imshowMy(f2,[])
title('分割线叠加在原始图像上的结果（用梯度和分水岭变换分割灰度图像）')
%% imregionalmin  imextendedmin 明白啦:）
clc
clear
A = 5*ones(10,10);
A(2:4,2:4) = 2; 
A(3,3) = 1; 
A(6:8,1:10) = 7; 
A
% A =
%     10    10    10    10    10    10    10    10    10    10
%     10     2     2     2    10    10    10    10    10    10
%     10     2     2     2    10    10    10    10    10    10
%     10     2     2     2    10    10    10    10    10    10
%     10    10    10    10    10    10    10    10    10    10
%     10    10    10    10    10     7     7     7    10    10
%     10    10    10    10    10     7     7     7    10    10
%     10    10    10    10    10     7     7     7    10    10
%     10    10    10    10    10    10    10    10    10    10
%     10    10    10    10    10    10    10    10    10    10

B = imregionalmin(A)
% B = 
%      0     0     0     0     0     0     0     0     0     0
%      0     1     1     1     0     0     0     0     0     0
%      0     1     1     1     0     0     0     0     0     0
%      0     1     1     1     0     0     0     0     0     0
%      0     0     0     0     0     0     0     0     0     0
%      0     0     0     0     0     1     1     1     0     0
%      0     0     0     0     0     1     1     1     0     0
%      0     0     0     0     0     1     1     1     0     0
%      0     0     0     0     0     0     0     0     0     0
%      0     0     0     0     0     0     0     0     0     0

BW0 = imextendedmin(A,0)
BW1 = imextendedmin(A,1)
BW2 = imextendedmin(A,2)
BW3 = imextendedmin(A,3)
BW4 = imextendedmin(A,4)
BW5 = imextendedmin(A,5)
BW6 = imextendedmin(A,6)
BW8 = imextendedmin(A,8)
BW9 = imextendedmin(A,9)
BW10 = imextendedmin(A,10)

%% 例10.12 用标记符控制的分水岭变换分割图像实例  其中 imextendedmin(f,2)
clc
clear
f = imread('..\Pictures\images_ch10\Fig1022(a)(gel-image).tif');
imshowMy(f)
title('原始凝胶图像')

h = fspecial('sobel');
fd = double(f);
g = sqrt(imfilter(fd,h,'replicate').^2 + imfilter(fd,h','replicate').^2);
imshowMy(g,[])
title('梯度幅度图像')

L = watershed(g);
wr = L == 0;
imshowMy(wr,[])
title('对梯度幅度图像进行分水岭变换（过分割）结果')

rm = imregionalmin(g); % 说明梯度图像有很多比较浅的坑: 造成的原因是原图像不均匀背景中的灰度细小变化造成的
imshowMy(rm,[])
title('对梯度幅度图像的局部最小区域')

im = imextendedmin(f,2); % 于是我们选择对原图像目标区域进行标记: 目标区域在原图像中就应该是一个比较深的坑
imshowMy(im,[])
title('对梯度幅度图像的局部最小区域（消除无关的小区域）')

fim = f;
fim(im) = 175; % 用灰度值175来标记目标区
imshowMy(fim,[])
title('内部标记符')

imshowMy(bwdist(im),[])
Lim = watershed(bwdist(im));
em = Lim == 0;
imshowMy(em,[])
title('外部标记符')

g2 = imimposemin(g,im | em);
imshowMy(g2,[])
title('修改后的梯度幅度图像')

L2 = watershed(g2);
f2 = f;
f2(L2 == 0) = 255;
imshowMy(f2,[])
title('用标记符控制的分水岭变换分割图像结果')

%% 例10.12 用标记符控制的分水岭变换分割图像实例  其中 imextendedmin(f,20)
clc
clear
f = imread('..\Pictures\images_ch10\Fig1022(a)(gel-image).tif');
imshowMy(f)
title('原始凝胶图像')

h = fspecial('sobel');
fd = double(f);
g = sqrt(imfilter(fd,h,'replicate').^2 + imfilter(fd,h','replicate').^2);
imshowMy(g,[])
title('梯度幅度图像')

L = watershed(g);
wr = L == 0;
imshowMy(wr,[])
title('对梯度幅度图像进行分水岭变换（过分割）结果')

rm = imregionalmin(g);
imshowMy(rm,[])
title('对梯度幅度图像的局部最小区域')

im = imextendedmin(f,20);
fim = f;
fim(im) = 175;
imshowMy(fim,[])
title('内部标记符')

Lim = watershed(bwdist(im));
em = Lim == 0;
imshowMy(em,[])
title('外部标记符')

g2 = imimposemin(g,im | em);
imshowMy(g2,[])
title('修改后的梯度幅度图像')

L2 = watershed(g2);
f2 = f;
f2(L2 == 0) = 255;
imshowMy(f2,[])
title('用标记符控制的分水岭变换分割图像结果')

%% 例10.12 用标记符控制的分水岭变换分割图像实例  其中 imextendedmin(f,iii) 自动化实现保存:)
clc
clear
f = imread('..\Pictures\images_ch10\Fig1022(a)(gel-image).tif');
% imshowMy(f)
% title('原始凝胶图像')

h = fspecial('sobel');
fd = double(f);
g = sqrt(imfilter(fd,h,'replicate').^2 + imfilter(fd,h','replicate').^2);
% imshowMy(g,[])
% title('梯度幅度图像')

L = watershed(g);
wr = L == 0;
% imshowMy(wr,[])
% title('对梯度幅度图像进行分水岭变换（过分割）结果')

rm = imregionalmin(g);
% imshowMy(rm,[])
% title('对梯度幅度图像的局部最小区域')

for i = 5:1:20
    im = imextendedmin(f,i);
    fim = f;
    fim(im) = 175;
    % imshowMy(fim,[])
    % title('内部标记符')

    Lim = watershed(bwdist(im));
    em = Lim == 0;
    % imshowMy(em,[])
    % title('外部标记符')

    g2 = imimposemin(g,im | em);
    % imshowMy(g2,[])
    % title('修改后的梯度幅度图像')

    L2 = watershed(g2);
    f2 = f;
    f2(L2 == 0) = 255;
    imshowMy(f2,[])
    title(['用标记符控制的分水岭变换分割图像结果 imextendedmin(f,' int2str(i) ')'])
    imwrite(f2, ['..\NewPictures\New_images_ch10\' '用标记符控制的分水岭变换分割图像结果 imextendedmin(f,' int2str(i) ')' '.jpg'])
    
end

%% watershed  2-D Example
clc
clear

% 2-D Example
% Make a binary image containing two overlapping circular objects. 
center1 = -10;
center2 = -center1;
dist = sqrt(2*(2*center1)^2);
radius = dist/2 * 1.4;
lims = [floor(center1-1.2*radius) ceil(center2+1.2*radius)];
[x,y] = meshgrid(lims(1):lims(2));
bw1 = sqrt((x-center1).^2 + (y-center1).^2) <= radius;
bw2 = sqrt((x-center2).^2 + (y-center2).^2) <= radius;
bw = bw1 | bw2;
imshowMy(bw,'n'), title('BW')
% Compute the distance transform of the complement of the binary image. 
D = bwdist(~bw);
imshowMy(D,[],'n'), title('Distance transform of ~bw')
% Complement the distance transform, and force pixels that don't belong to the objects to be at -Inf. 
D = -D;
D(~bw) = -Inf;
% Compute the watershed transform and display it as an indexed image. 
L = watershed(D);
rgb = label2rgb(L,'jet',[.5 .5 .5]);
imshowMy(rgb,'n'), title('Watershed transform of D');




%% watershed  3-D Example
clc
clear

% 3-D Example
% Make a 3-D binary image containing two overlapping spheres. 
center1 = -10;
center2 = -center1;
dist = sqrt(3*(2*center1)^2);
radius = dist/2 * 1.4;
lims = [floor(center1-1.2*radius) ceil(center2+1.2*radius)];
[x,y,z] = meshgrid(lims(1):lims(2));
bw1 = sqrt((x-center1).^2 + (y-center1).^2 + ...
     (z-center1).^2) <= radius;
bw2 = sqrt((x-center2).^2 + (y-center2).^2 + ...
     (z-center2).^2) <= radius;
bw = bw1 | bw2;
figure, isosurface(x,y,z,bw,0.5), axis equal, title('BW')
xlabel x, ylabel y, zlabel z
xlim(lims), ylim(lims), zlim(lims)
view(3), camlight, lighting gouraud
% Compute the distance transform. 
D = bwdist(~bw);
figure, isosurface(x,y,z,D,radius/2), axis equal
title('Isosurface of distance transform')
xlabel x, ylabel y, zlabel z
xlim(lims), ylim(lims), zlim(lims)
view(3), camlight, lighting gouraud
% Complement the distance transform, force nonobject pixels to be -Inf, and then compute the watershed transform. 
D = -D;
D(~bw) = -Inf;
L = watershed(D);
figure, isosurface(x,y,z,L==2,0.5), axis equal
title('Segmented object')
xlabel x, ylabel y, zlabel z
xlim(lims), ylim(lims), zlim(lims)
view(3), camlight, lighting gouraud
figure, isosurface(x,y,z,L==3,0.5), axis equal
title('Segmented object')
xlabel x, ylabel y, zlabel z
xlim(lims), ylim(lims), zlim(lims)
view(3), camlight, lighting gouraud


%% 图像分割大总结

1、 分水岭分割方法的主要优点是：提出一种能有效使用“先验知识”的机制。典型应用就是“使用控制标记符的分水岭分隔”
    该进方案 A ：利用此方法分隔之后只保留平均灰度介于 [A B] 之间的分隔圈




























%% 
clc
clear


《红楼梦·第一回》(好了歌)
       -- 清·曹雪芹
陋室空堂,当年笏满床.衰草枯杨,曾为歌舞场.
蛛丝儿结满雕梁,绿纱今又糊在蓬窗上.
说什么脂正浓,粉正香,如何两鬓又成霜 
昨日黄土陇头送白骨,今宵红灯帐底卧鸳鸯.
金满箱,银满箱,展眼乞丐人皆谤.
正叹他人命不长,那知自己归来丧!
训有方,保不定日后作强梁.择膏粱,谁承望流落在烟花巷!
因嫌纱帽小,致使锁枷杠.昨怜破袄寒,今嫌紫蟒长.
乱烘烘你方唱罢我登场,反认他乡是故乡.
甚荒唐,到头来都是为他人作嫁衣裳!






