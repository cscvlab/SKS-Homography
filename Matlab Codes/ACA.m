%% A MATLAB code of Cai's homography solver ACA [1].
% This software will only be used for non-commercial research purposes. See License.md.
% For commercial use, please contact the authors.

%  Affine-Core-Affine (ACA) Decomposition of 2D homographies:
%
%              H = H_A2^(-1) * H_C * H_A1
%
%  FLOPs: 97 = 85 (compute homographies up to a scale from 4 point correspondences) + 12 (normalization)

%  Detailed explanation goes here
%  Inputs: SourcePts --- source points; 
%              TargetPts --- target points;   
%              Both of them (3*4 matrices) are represented in homogeneous coordinates.
%  Outputs: H --- homography between source and target planes  with normalization of the last element;

% REFERENCE:
%   [1] S. Cai, et al., "Fast and Interpretable 2D Homography Decomposition: Similarity-Kernel-Similarity (SKS) 
%        and Affine-Core-Affine (ACA)", submitted.


function [ H ] = ACA ( SourcePts, TargetPts )

    % compute the affine trans. H_A1 on source plane based on any three anchor points (e.g., M1N1P1),  7 flops                                           
    M1 = SourcePts(1:2, 1);   % 2D coordinates
    N1 = SourcePts(1:2, 2);
    P1 = SourcePts(1:2,3);
    Q1 = SourcePts(1:2,4);
    M1N1 = N1 - M1;          % vector M1N1, 2 flops
    M1P1 = P1 - M1;           % vector M1P1, 2 flops
    f_A1 = M1N1(1)*M1P1(2)-M1N1(2)*M1P1(1);       % 3 flops
    % matrix representation, no need in C++ procedure
    H_A1_T = [1 0 -M1(1); 0 1 -M1(2); 0 0 1];     % translation matrix in H_A1 
    H_A1_U = [M1P1(2) -M1P1(1) 0; -M1N1(2) M1N1(1) 0; 0 0  f_A1];        % upper-left affine components in H_A1
    H_A1 =  H_A1_U*H_A1_T;        % affine transformation on source plane
    %              [ M1P1(2)   -M1P1(1)               ]         [1        -M1(1)]
    % H_A1 = [ -M1N1(2)  M1N1(1)               ]    *   [     1   -M1(2)]
    %              [                                       f_A1  ]         [               1    ]
    
    % compute H_A2 on target plane by M2N2P2, 7 flops
    M2 = TargetPts(1:2,1);
    N2 = TargetPts(1:2,2);
    P2 = TargetPts(1:2,3);
    Q2 = TargetPts(1:2,4);
    M2N2 = N2 - M2;          % 2 flops
    M2P2 = P2 - M2;           % 2 flops
    f_A2 = M2N2(1)*M2P2(2)-M2N2(2)*M2P2(1);       % 3 flops 
    % matrix representation, no need in C++ procedure
    H_A2_T = [1 0 -M2(1); 0 1 -M2(2); 0 0 1];     % translation matrix in H_A2 
    H_A2_U = [M2P2(2) -M2P2(1) 0; -M2N2(2) M2N2(1) 0; 0 0 f_A2];        % upper-left affine components in H_A2 
    H_A2 = H_A2_U*H_A2_T;                               % affine transformation on target plane
    H_A2_inv = [ M2N2(1)   M2P2(1)    M2(1); M2N2(2)   M2P2(2)    M2(2); 0 0 1 ];
    
    % substitute the fourth point to solve H_C with 2 DOF,  totally 28 flops
    %            [ a+b    0     0  ]         [1 0 0]     [ 1+b/a      0      0  ]        [    C_11                  0              0   ]
    % H_C = [    0    a+v   0 ]  = a* [0 1 0]  * [     0      1+v/a   0  ]   =  [       0                  C_22           0   ]
    %            [    b     v     a  ]          [1  1 1]   [    -1         -1      1  ]       [C_11-C_33     C_22 - C_33   C_33]
    % Q3 = H_A1_U*H_A1_T*Q1;   8 flops
    % Q4 = H_A2_U*H_A2_T*Q2;   8 flops
    M1Q1 = Q1 - M1;          % 2 flops
    M2Q2 = Q2 - M2;          % 2 flops
    Q3(1) = M1P1(2) * M1Q1(1) - M1P1(1) * M1Q1(2);          % 3 flops
    Q3(2) = M1N1(1) * M1Q1(2) - M1N1(2) * M1Q1(1);         % 3 
    Q4(1) = M2P2(2) * M2Q2(1) - M2P2(1) * M2Q2(2);          % 3
    Q4(2) = M2N2(1) * M2Q2(2) - M2N2(2) * M2Q2(1);         % 3
    tt1 = f_A1 - Q3(1) - Q3(2);       % 2 flops
    tt2 = f_A2 - Q4(1) - Q4(2);       % 2 flops
    C_11 = Q4(1) * Q3(2) * tt1;      % 2 flops
    C_22 = Q4(2) * Q3(1) * tt1;      % 2 flops
    C_33 = Q3(1) * Q3(2) * tt2;      % 2 flops
    C_31 = C_11 - C_33;        % 1 flops
    C_32 = C_22 - C_33;        % 1 flops
    % matrix representation, no need in C++ procedure
    H_C = [C_11    0       0; 
                   0    C_22    0; 
                C_31 C_32 C_33];  
    
    % compute homography up to a scale by matrices multiplication. 43 flops
    H = H_A2_inv*H_C*H_A1_U*H_A1_T;        
    % express H1 = H_A2_inv * H_C, actually has 10 flops
    % only compute the 2*2 upper-left elements. C_33*M2 are repeated twice. no need to compute the last row
	%			 [ C_11*N2(1)-C_33*M2(1)     C_22*P2(1)-C_33*M2(1)     C_33*M2(1)  ]
	% H1 =   [ C_11*N2(2)-C_33*M2(2)     C_22*P2(2)-C_33*M2(2)     C_33*M2(2)  ]
	%			 [                 C_31                                   C_32                          C_33       ]
	%		
	% the multiplication H2 = H1 * H_A1_U has 18 flops. only compute the 3*2 left elements. no need to compute the last column
    %			 [ H1_11*M1P1(2)-H1_12*M1N1(2)     H1_12*M1N1(1)-H1_11*M1P1(1)      f_A1*C_33*M2(1)  ]
	% H2 =   [ H1_21*M1P1(2)-H1_22*M1N1(2)     H1_22*M1N1(1)-H1_21*M1P1(1)      f_A1*C_33*M2(2)  ]
	%			 [   C_31*M1P1(2)-C_32*M1N1(2)          C_32*M1N1(1)-C_31*M1P1(1)              f_A1*C_33       ]
    %
    % the multiplication H = H2 * H_A1_T has 15 flops. C_33*M2 has been computed
    %			 [ H2_11    H2_12      f_A1*C_33*M2(1)-H2_11*M1(1)-H2_12*M1(2)  ]
	%  H  =   [ H2_21    H2_22      f_A1*C_33*M2(2)-H2_21*M1(1)-H2_22*M1(2)  ]
	%			 [ H2_31    H2_32             f_A1*C_33-H2_31*M1(1)-H2_32*M1(2)      ]
    
    % if necessary, normalization based on the last element of H, 12 flops
    H =H ./ H(3,3);
    
end

