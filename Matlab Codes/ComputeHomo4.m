% 
% 2019-09-04 hammer
% rewrite the function 'ComputeHomo4' to compute homography for 4-point case
% which decompose H into: H = inv(H_S2) * H_K * H_S1. Here the kernel
% transformation H_K is computed by the first way named SKS^{I}, which is
% implemented by the function 'CompKernelFrom4'.
%
% 2021-07-08 hammer
% add the  second way to compute homography for 4-point case.  Here the kernel
% transformation H_K_2 is computed by SKS^{II}, which is implemented by the 
% function 'CompKernelFrom4_2'. Thus H_2 = inv(H_S2) * H_K_2 * H_S1.
%

%   Detailed explanation goes here
%   Inputs: SourcePts---source points; TargetPts---target points;   
%           Both of them (3*4 matrices) are represented in homogeneous coordinates.
%           If the number of points is greater than 4, please call another funciton
%           ComputeHomoN specially designed for n-point case.
%   Outputs: H_1 & H_2 --- homographies computed by two methods between source and target planes;


function [ H_1, H_2 ] = ComputeHomo4( SourcePts, TargetPts )

    % compute H_S1 on source plane  by TAP, totally 15 flops (excluding the expression of H_S1)                                            
    M1 = SourcePts(:,1);
    N1 = SourcePts(:,2);
    O1 = 0.5*(M1+N1);    % midpoint, 4 flops
    ON1 = N1 - O1;          % vector ON1, 2 flops
    k1 = 1/sum(ON1.^2);     % 7 flops (division is condisered as 4 flops)
    a1 = k1*ON1(1);             % 1 flops
    b1 = -k1*ON1(2);           % 1 flops
    H_T_1 = [1 0 -O1(1); 0 1 -O1(2); 0 0 1];     % translation matrix in H_S1 
    H_AB_1 = [a1 -b1 0; b1 a1 0; 0 0 1];          % scale and rotation matrix in H_S1 
    H_S1 = H_AB_1*H_T_1;                               % similar transformation on source plane
    % 7 flops
    %              [ a1   -b1    b1*O1(2)-a1*O1(1) ]
    % H_S1 = [ b1    a1   -b1*O1(1)-a1*O1(2) ]
    %              [                             1                  ]
    
    % compute H_S2 on target plane  by TAP, totally 15 flops
    M2 = TargetPts(:,1);
    N2 = TargetPts(:,2);
    O2 = 0.5*(M2+N2);    % midpoint, 4 flops
    ON2 = N2 - O2;          % vector ON2, 2 flops
    k2 = 1/sum(ON2.^2);     % 7 flops
    a2 = k2*ON2(1);             % 1 flops
    b2 = -k2*ON2(2);           % 1 flops
    H_T_2 = [1 0 -O2(1); 0 1 -O2(2); 0 0 1];     
    H_AB_2 = [a2 -b2 0; b2 a2 0; 0 0 1];
    H_S2 = H_AB_2*H_T_2;                               % similar transformation on target plane
    H_S2_inv = [ON2(1) -ON2(2) O2(1); ON2(2) ON2(1) O2(2); 0 0 1];    % inverse matrix of H_S2
    
	% call function of computing kernel transformation H_K
    PQ1 = [SourcePts(:,3) SourcePts(:,4)];
    PQ2 = [TargetPts(:,3) TargetPts(:,4)];
    H_K = CompKernelFrom4(H_S1, H_S2, PQ1, PQ2);             % H_K--kernel transformation, 1st solution, totally 79 flops
    H_K_2 = CompKernelFrom4_2(H_S1, H_S2, PQ1, PQ2);      % H_K_2--kernel transformation, 2nd solution, totally 75 flops
    
    % final homography (1st solution)
    H_1 = H_S2_inv*H_K*H_S1;
    % express H_L = H_S2^(-1) * H_K
	% actually has 20 flops
	%			 [ a_K*ON2(1)+b_K*O2(1)     u_K*ON2(1)+v_K*O2(1)-ON2(2)    a_K*O2(1)+b_K*ON2(1)  ]
	% H_L =  [ a_K*ON2(2)+b_K*O2(2)      u_K*ON2(2)+v_K*O2(2)+ON2(1)   a_K*O2(2)+b_K*ON2(2) ]
	%			 [                b_K                                             v_K                                              a_K              ]
	%		
	% the multiplication H_1 = H_L * H_S1 actually has 30 flops, as there are three special elements in  H_S1
    
    % final homography (2nd solution)
    H_2 = H_S2_inv*H_K_2*H_S1;
    % similarly, express H_L_2 = H_S2^(-1) * H_K_2 actually has 22 flops
	% the multiplication H_2 = H_L_2 * H_S1 still has 30 flops
    
end

