%
% 2019-09-04 hammer
% write the function 'CompKernelFromN' to compute kernel transformation H_K under N-point case
% this method mainly utilizes the cross-ratio invariants of concurrent lines 
% intersecting at TAP imposed by the third point correspondence 
%

%   Detailed explanation goes here
%   Inputs: X_3---N-2 normalized source points;    X_4---N-2 normalized target points;   
%           Both of them (3*(N-2) matrices) are in homogeneous coordinates
%           If the number of points is equal to 4, please call another funciton
%           'CompKernelFrom4' or 'CompKernelFrom4_2'  specially designed for 4-point case
%   Outputs: H_K--kernel transformation between normalized source and target planes;


function [ H_K ] = CompKernelFromN( X3, X4 )
    
    % remove
    % X1 = PQ_src;
    % X3 = H_S_src*X1;
    % X2 = PQ_tar;
    % X4 = H_S_tar*X2;

    % initialize
    Exx1 = 0;
    Exy1 = 0;
    Eyy1 = 0;
    Exz1 = 0;
    Eyz1 = 0;
    Exx2 = 0;
    Exy2 = 0;
    %Eyy2 = 0;   % Eyy1=Eyy2
    Exz2 = 0;
    Eyz2 = 0;

    % process every point to obtain coefficients, (N-2)*27 flops
    for i=1:length(X3(1,:))
        q1_x = X3(1,i);
        q1_y = X3(2,i);
        q2_x = X4(1,i);
        q2_y = X4(2,i);
        x1 = (q1_x + 1) * q2_y;
        x2 = (q1_x - 1) * q2_y;
        y1 = q1_y * q2_y;
        % y2 = q1_y * q2_y;     % y1=y2
        z1 = -(q2_x + 1) * q1_y;
        z2 = -(q2_x - 1) * q1_y;

        Exx1 = Exx1 + x1 * x1;    % sum of  x1*x1
        Exy1 = Exy1 + x1 * y1;
        Eyy1 = Eyy1 + y1 * y1;
        Exz1 = Exz1 + x1 * z1;
        Eyz1 = Eyz1 + y1 * z1;

        Exx2 = Exx2 + x2 * x2;
        Exy2 = Exy2 + x2 * y1;
        %Eyy2 = Exx1 + y2 * y2;     % Eyy1=Eyy2
        Exz2 = Exz2 + x2 * z2;
        Eyz2 = Eyz2 + y1 * z2;
    end    

    % optimal solution of loss function sum(x1*aab+y1*uav+z1)^2, 15 flops
    fenmu = 1 / (Exy1*Exy1 - Exx1*Eyy1);
    aab = fenmu * (Exz1*Eyy1 - Exy1*Eyz1);       %  aab: a_K+b_K
    uav = fenmu * (Exx1*Eyz1 - Exy1*Exz1);       %  uav: u_K+v_K
    % optimal solution of loss function sum(x2*aab+y1*uav+z2)^2, 15 flops
    fenmu = 1 / (Exy2*Exy2 - Exx2*Eyy1);
    amb = fenmu * (Exz2*Eyy1 - Exy2*Eyz2);     %  amb: a_K-b_K
    umv = fenmu * (Exx2*Eyz2 - Exy2*Exz2);      %  umv: u_K-v_K

    % compute a, b, u, v, 8 flops
    a_K = 0.5 * (aab + amb);
    b_K = 0.5 * (aab - amb);
    u_K = 0.5 * (uav + umv);
    v_K = 0.5 * (uav - umv);

    H_K = [a_K u_K b_K; 0 1 0; b_K v_K a_K];

end

