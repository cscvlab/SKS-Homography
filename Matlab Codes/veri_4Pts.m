%
% Simple verification of the correctness of SKS and ACA
%

clear all 
close all

%% 1 Source plane configuration
M = [0, 0, 1]';    % manually set 4 points
N = [200, 0, 1]';
P = [50, 139, 1]';
Q = [181, 93, 1]';
SrcPts = [M N P Q];
np = 4;
figure, scatter(SrcPts(1,:), SrcPts(2,:))   % visualiztion
hold on;
set(gca,'YDir','reverse');
set(gca,'XAxisLocation', 'top');
box on
title('source plane');
for ii = 1:np
    c = num2str(ii);
    text(SrcPts(1,ii)+3, SrcPts(2,ii), c);     % label every point
end

%% 2 Camera parameters setting
% intrinsic parameters
fu = 900;
fv = 900;
u0 = 500;
v0 = 400;
K = [fu 0 u0;0 fv v0;0 0 1];

% image size
nx = 1024;            
ny = 768;

% fixed extrinsic parameters
r_x = -pi/8.*sqrt(5);
r_y = -pi/8.*sqrt(5);
r_z = -pi/16.*sqrt(5);
R_x = [ 1 0 0;0 cos(r_x) -sin(r_x);0 sin(r_x) cos(r_x)];
R_y = [ cos(r_y) 0 sin(r_y);0 1 0;-sin(r_y) 0 cos(r_y)];
R_z = [ cos(r_z) -sin(r_z) 0;sin(r_z) cos(r_z) 0;0 0 1];
R = R_x * R_y * R_z;
T = [-10.5 -12.5 525]';


%% 3 Projection
RT = [R(:,[1 2]), T];
H_real = K * RT;
TarPts = H_real * SrcPts;
TarPts = [TarPts(1,:)./TarPts(3,:); TarPts(2,:)./TarPts(3,:); ones(1,length(TarPts))];
noise1 = 0;
TarPts(1:2,:) =  TarPts(1:2,:) + noise1;     % add noise if necessary
figure, 
set(gca,'YDir','reverse');
set(gca,'XAxisLocation', 'top');
box on
hold on
scatter(TarPts(1,:),TarPts(2,:))
title('image');
axis([0 1024 0 768]);
for ii=1:np
    c=num2str(ii);
    text(TarPts(1,ii)+9,TarPts(2,ii),c);     % label every point
end


%% 4 Compute homography utilizing SKS and ACA
H_SKS = SKS( SrcPts, TarPts );     
% test
ratio1 = H_real./ H_SKS;

H_ACA = ACA( SrcPts, TarPts );     
% test
ratio2 = H_real./ H_ACA;


%% 5 Compute homography for a given rectangle
% prepare data of rectangle
width = 50;
height = 40;
ratio_rec = width / height;
M_x = 36;       % upper-left point M
M_y = 81;
N_x = M_x + width;
P_y = M_y + height;
SrcPts = [ M_x N_x M_x N_x; M_y M_y P_y P_y; 1 1 1 1];   % order: MNPQ
TarPts = H_real * SrcPts;
TarPts = [TarPts(1,:)./TarPts(3,:); TarPts(2,:)./TarPts(3,:); ones(1,length(TarPts))];

H_ACA_rect = ACA_rect( TarPts, M_x, M_y, width, ratio_rec );
% test
ratio3 = H_real./ H_ACA_rect;


