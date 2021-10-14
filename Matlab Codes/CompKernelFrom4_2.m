%
% 2021-07-08 hammer
% write the function 'CompKernelFrom4_2' to compute kernel
% transformation H_K under 4-point case with the second solving process.
% Here H_K is decomposed into three sub-kernel transformations. i.e.,
%
%   H_K = inv(H_K2)*H_KC*H_K1;  
%

%   Detailed explanation goes here
%   Inputs: PQ_src--two source points;    PQ_tar--two target points;   
%           Both of them (3*2 matrices) are in homogeneous coordinates.
%           H_S_src--similar transformation on source plane
%           H_S_tar--similar transformation on target plane
%           If the number of points is greater than 4, please call another funciton
%           'CompKernelFromN' specially designed for optimal estimation under n-point case.
%   Outputs: H_K--kernel transformation between normalized source and target planes;
%   FLOPS: 4*8+2*8+12+15 = 75


function [ H_K ] = CompKernelFrom4_2( H_S_src, H_S_tar, PQ_src, PQ_tar )
    
    % obtain points under similarity transformations
    P1 = PQ_src(:,1);
    P3 = H_S_src*P1;     % 8 flops
    P2 = PQ_tar(:,1);
    P4 = H_S_tar*P2;     % 8 flops
    Q1 = PQ_src(:,2);
    Q3 = H_S_src*Q1;    % 8 flops
    Q2 = PQ_tar(:,2);
    Q4 = H_S_tar*Q2;    % 8 flops

    % obtain sub-kernel transformation H_K1
    t_K1 = (P3(2)-Q3(2));                  % 1 flops
    nu_K1 = (P3(1)-Q3(1)) / t_K1;     % 5 flops
    mu_K1 = P3(1) - nu_K1*P3(2);    % 2 flops

    % obtain sub-kernel transformation H_K2
    t_K2 = 1 / (P4(2)-Q4(2));             % 5 flops
    nu_K2 = t_K2*(P4(1)-Q4(1));       % 1 flops
    mu_K2 = P4(1) - nu_K2*P4(2);    % 2 flops

    % obtain sub-kernel transformation H_KC
    t_KC = t_K2 / (P3(2)*Q3(2));                               % 5 flops
    rec_a_KC = t_K1*t_KC*(P4(2)*Q4(2));                  % 3 flops     reciprocal of  a_KC
    new_v_KC = t_KC*(P3(2)*Q4(2)-Q3(2)*P4(2));    % 4 flops     new_v_KC = v_KC/a_KC

    % compute kernel transformation H_K
    K_11 = 1 - mu_K1*mu_K2;      % 2 flops
    K_13 = mu_K2 - mu_K1;          % 1 flops
    temp = 1 - mu_K1*mu_K1;     % 2 flops
    K_22 = rec_a_KC*temp;           % 1 flops
    K_12 = K_22*nu_K2 + new_v_KC*mu_K2*temp - nu_K1*K_11;     % 6 flops
    K_32 = new_v_KC*temp - nu_K1*K_13;      % 3 flops

    H_K = [K_11 K_12 K_13; 0 K_22 0; K_13 K_32 K_11];

    % % only for test
    % H_K1 = [1 -nu_K1 -mu_K1; 0 1-mu_K1^2 0; -mu_K1 mu_K1*nu_K1 1];
    % inv_H_K2 = [1 nu_K2 mu_K2; 0 1 0; mu_K2 0 1];
    % H_KC = [1 0 0; 0 inv_a_KC 0; 0 new_v_KC 1];
    % P4_veri = H_KC*[0 P3(2) 1]';
    % adg1 = P4_veri(2)/P4_veri(3) - P4(2);
    % Q4_veri = H_KC*[0 Q3(2) 1]';
    % adg2 = Q4_veri(2)/Q4_veri(3) - Q4(2);
    % H_K_veri = inv_H_K2*H_KC*H_K1;
    % fdsa = H_K_veri./H_K;   % test

end

