% hammer 210411
% test SKS-I, SKS-II and SKS-n with random points.

clear all 
close all

%% 1 Source Plane Configuration
model_width = 400;
model_height = 300;
np = 100;
xr = model_width * ( rand(np,1) - 0.5 );
yr = model_height * ( rand(np,1) - 0.5 );
figure, scatter(xr,yr)   % visualiztion
hold on;
set(gca,'YDir','reverse');
set(gca,'XAxisLocation', 'top');
box on
title('random points in object plane');
for ii=1:np
    c=num2str(ii);
    text(xr(ii)+4,yr(ii),c)   % label every point
end
axis([-0.5*model_width 0.5*model_width -0.5*model_height 0.5*model_height]);
SourcePts = [xr';yr';ones(1,np)];   % homogeneous coordinates


%% 2 Camera Parameters Setting
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
r_x = -pi/6/sqrt(5);
r_y = -pi/6/sqrt(5);
r_z = -pi/12/sqrt(5);
R_x = [ 1 0 0;0 cos(r_x) -sin(r_x);0 sin(r_x) cos(r_x)];
R_y = [ cos(r_y) 0 sin(r_y);0 1 0;-sin(r_y) 0 cos(r_y)];
R_z = [ cos(r_z) -sin(r_z) 0;sin(r_z) cos(r_z) 0;0 0 1];
R2 = R_x * R_y * R_z;
T2 = [-10.5 -12.5 525]';


%% 3 Projection
R = R2;
T = T2;
Proj1 = K*[R T];
RT = [R(:,[1 2]), T];
H_real = K*RT;
TarPts = H_real*SourcePts;
TarPts = [TarPts(1,:)./TarPts(3,:); TarPts(2,:)./TarPts(3,:); ones(1,length(TarPts))];
noise1 = 0*randn(2,np);
%noise1 = 0.5*randn(2,np);
TarPts(1:2,:) =  TarPts(1:2,:) + noise1;     % add noise if necessary
figure, 
set(gca,'YDir','reverse');
set(gca,'XAxisLocation', 'top');
box on
hold on
scatter(TarPts(1,:),TarPts(2,:))
title('image plane');
axis([0 nx 0 ny]);
for ii=1:np
    c=num2str(ii);
    text(TarPts(1,ii)+3,TarPts(2,ii),c)   % label every point
end


%% 4 Compute Homography with 4 Points
[H_1, H_2] = ComputeHomo4( SourcePts(:,1:4), TarPts(:,1:4) );     % two ways to compute homography
% test
dfd = H_real./ H_1;
dfd2 = H_real./ H_2;


%% 5 Computing Homography with N Points
% compute homography using SKS method
[H_est, H_opt] = ComputeHomoN( SourcePts, TarPts );
dsd = H_real./ H_est;     %test  
dsd2 = H_real./ H_opt;
TarPts_est = H_est*SourcePts;       % reprojection points
TarPts_est = TarPts_est ./ (ones(3,1)*TarPts_est(3,:));
err = TarPts_est - TarPts;    % reprojection error
repro = zeros(np,1);
for ii=1:np
    repro(ii) = sqrt(err(1,ii)^2 + err(2,ii)^2);
end
mean_err_SKS= sum(repro)/np;     % mean reprojection error of this trial

% H_opt
TarPts_est = H_opt*SourcePts;      % reprojection points
TarPts_est = TarPts_est ./ (ones(3,1)*TarPts_est(3,:));
err = TarPts_est - TarPts;     % reprojection error
repro = zeros(np,1);
for ii=1:np
    repro(ii) = sqrt(err(1,ii)^2 + err(2,ii)^2);
end
mean_err_SKS_opt = sum(repro)/np;

        

