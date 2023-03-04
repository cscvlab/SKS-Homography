%% A MATLAB code of Cai's homography solver SKS [1].
% This software will only be used for non-commercial research purposes. See License.md.
% For commercial use, please contact the authors.

%  Similarity-Kernel-Similarity (SKS) Decomposition of 2D homographies:
%
%              H = H_S2^(-1) * H_K * H_S1 = H_S2^(-1) * H_E * H_T2_inv * H_G * H_T1 * H_E * H_S1
%
%  FLOPs: 169 = 157 (compute homographies up to a scale from 4 point correspondences) + 12 (normalization)

%  Detailed explanation goes here
%  Inputs: SourcePts --- source points; 
%              TargetPts --- target points;   
%              Both of them (3*4 matrices) are represented in homogeneous coordinates with the last elements 1.
%  Outputs: H --- homography between source and target planes  with normalization of the last element;

% REFERENCE:
%   [1] S. Cai, et al., "Fast and Interpretable 2D Homography Decomposition: Similarity-Kernel-Similarity (SKS) 
%        and Affine-Core-Affine (ACA)", submitted.


function [ H ] = SKS ( SourcePts, TargetPts )

    % compute the similarity transformation H_S1 on source plane based on two anchor points (e.g., M1 and N1), 9 flops                                        
    M1 = SourcePts(1:2, 1);   % 2D coordinates
    N1 = SourcePts(1:2, 2);
    O1 = 0.5 * (M1 + N1);    % midpoint, 4 flops
    ON1 = N1 - O1;              % vector ON1, 2 flops
    f_S1 = sum(ON1.^2);      % sqaure of length of ON1, 3 flops 
    % matrix representation, no need in C++ procedure
    H_S1_T = [1 0 -O1(1); 0 1 -O1(2); 0 0 1];     % translation matrix in H_S1 
    H_S1_SR = [ON1(1) ON1(2) 0; -ON1(2) ON1(1) 0; 0 0 f_S1];          % scale and rotation matrix in H_S1 
    H_S1 = H_S1_SR * H_S1_T ;                          % similar transformation on source plane
    %              [ ON1(1)     ON1(2)    -ON1(1)*O1(1)-ON1(2)*O1(2) ]
    % H_S1 = [ -ON1(2)    ON1(1)     ON1(1)*O1(2)-ON1(2)*O1(1) ]
    %              [                                                         f_S1                  ]
        
    % compute H_S2 on target plane based on TAP, 9 flops
    M2 = TargetPts(1:2, 1);
    N2 = TargetPts(1:2, 2);
    O2 = 0.5*(M2+N2);        % midpoint, 4 flops
    ON2 = N2 - O2;              % vector ON2, 2 flops
    f_S2 = sum(ON2.^2);      % sqaure of length of ON2, 3 flops 
    % matrix representation, no need in C++ procedure
    H_S2_T = [1 0 -O2(1); 0 1 -O2(2); 0 0 1];      % translation matrix in H_S2 
    H_S2_SR= [ON2(1) ON2(2) 0; -ON2(2) ON2(1) 0; 0 0 f_S2];       % scale and rotation matrix in H_S2 
    H_S2 = H_S2_SR * H_S2_T ;                            % similar transformation on target plane
    H_S2_inv = [ON2(1) -ON2(2) O2(1); ON2(2) ON2(1) O2(2); 0 0 1];    % inverse matrix of H_S2
    
	% compute P3 (8 flops) and P5 (6 flops)
    % 
    %									                 [ 1       ]   [  ON1(1)     ON1(2)         ]    [ 1     -O1(1)  ]    [ P1(1) ]     [ P5(1) ]
	% P5 = H_E*H_S1*P1 = H_E*P3  = [       1 ] * [ - ON1(2)   ON1(1)          ] * [    1  -O1(1)  ] * [ P1(2) ] = [ P5(2) ]
	%									                 [    1    ]   [		                        f_S1 ]    [             1     ]     [   1    ]     [   1     ]
    P1 = SourcePts(1:2, 3);
    OP1 = P1 - O1;      %  2 flops 
    P3(1) = ON1(1)*OP1(1) + ON1(2)*OP1(2);     % 3 flops
    P3(2) = ON1(1)*OP1(2) - ON1(2)*OP1(1);     % 3 flops
    % P3(3) = f_S1
    temp1 = 1 / P3(2);      % 4 flops
    P5(1) = temp1 * P3(1);      % 1 flops
    P5(2) = temp1 * f_S1;        % 1 flops
	
    % compute P4 (8 flops) and P6 (6 flops)
    P2 = TargetPts(1:2, 3);
    OP2 = P2 - O2;      % 2 flops 
    P4(1) = ON2(1)*OP2(1) + ON2(2)*OP2(2);     % 3 flops
    P4(2) = ON2(1)*OP2(2) - ON2(2)*OP2(1);     % 3 flops
    % P4(3) = f_S2
    temp2 = 1 / P4(2);      % 4 flops
    P6(1) = temp2 * P4(1);      % 1 flops
    P6(2) = temp2 * f_S2;        % 1 flops
    
    % compute Q3 (8 flops) and Q7 (6 flops)
    Q1 = SourcePts(1:2, 4);
    OQ1 = Q1 - O1;     % 2 flops
    Q3(1) = ON1(1)*OQ1(1) + ON1(2)*OQ1(2);     % 3 flops
    Q3(2) = ON1(1)*OQ1(2) - ON1(2)*OQ1(1);      % 3 flops
    % Q3(3) = f_S1
    % Q5 = [Q3(1) f_S1 Q3(2) ]';
    %                             [ P3(2)   0    -P3(1) ]              [ P3(2)*Q3(1) - P3(1)*Q3(2) ]
    % Q7 = H_T1*Q5 = [  0     P3(2)   -f_S1 ] * Q5 =  [    ( P3(2) - Q3(2) ) * f_S1    ]
    %                             [  0        0      P3(2) ]              [             P3(2) * Q3(2)         ]
    Q7 = [ P3(2)*Q3(1) - P3(1)*Q3(2); ( P3(2) - Q3(2) ) * f_S1; P3(2) * Q3(2) ];      % 6 flops
    
    % compute Q4 (8 flops) and Q8 (6 flops)
    Q2 = TargetPts(1:2, 4);
    OQ2 = Q2 - O2;     % 2 flops
    Q4(1) = ON2(1)*OQ2(1) + ON2(2)*OQ2(2);     % 3 flops
    Q4(2) = ON2(1)*OQ2(2) - ON2(2)*OQ2(1);      % 3 flops
    % Q4(3) = f_S2
    % Q6 = [Q4(1) f_S2 Q4(2) ]';
    Q8 = [ P4(2)*Q4(1) - P4(1)*Q4(2); ( P4(2) - Q4(2) ) * f_S2; P4(2) * Q4(2) ];      % 6 flops
    
    % solving the four variables in H_K (24 flops)  
    %                             [ a_K   b_K    0 ]             
    %   Q8 = H_G*Q7 = [ b_K   a_K   0 ] * Q7 
    %                             [   0       0     1 ]        
    % consequently, we have  
    %       Q7(1)*a_K + Q7(2)*b_K = Q8(1)*Q7(3)/Q8(3);
    %       Q7(1)*b_K + Q7(2)*a_K = Q8(2)*Q7(3)/Q8(3);
    % then, a_K and b_K can be obtained.
    temp1 = Q7(1)*Q8(1) - Q7(2)*Q8(2);      % 3 flops
    temp2 = Q7(1)*Q8(2) - Q7(2)*Q8(1);      % 3 flops
    temp3 = Q7(1)*Q7(1) - Q7(2)*Q7(2);      % 3 flops
    temp3 = Q7(3) / ( Q8(3)*temp3 );           % 5 flops
    a_K = temp1 * temp3;        % 1 flops
    b_K = temp2 * temp3;        % 1 flops
    u_K = P6(1) - P5(1)*a_K - P5(2)*b_K;       % 4 flops
    v_K = P6(2) - P5(1)*b_K - P5(2)*a_K;       % 4 flops
    % matrix representation, no need in C++ procedure
    H_K = [a_K u_K b_K; 0 1 0; b_K v_K a_K];
    
    % homography up to a scale in matrix form
    % 59 flops, H_S2_inv*H_K=20; H_S1=6; *H_S1=33.
    H = H_S2_inv*H_K*H_S1;         
    % first compute H_L = H_S2_inv * H_K   
    % 20 flops = 4*3+2*4
	%			 [ ON2(1)*a_K+O2(1)*b_K      ON2(1)*u_K+O2(1)*v_K-ON2(2)     ON2(1)*b_K+O2(1)*a_K  ]
	% H_L =  [ ON2(2)*a_K+O2(2)*b_K      ON2(2)*u_K+O2(2)*v_K+ON2(1)    ON2(2)*b_K+O2(2)*a_K  ]
	%			 [                b_K                                              v_K                                              a_K               ]
	%		
    % then compute the two elements in the last column of H_S1 (shown above) will cost 6 flops
    % the final multiplication H = H_L * H_S1 actually has 33 flops, as there are two zero elements in  H_S1
    
    % if necessary, normalization based on the last element of H, 12 flops
    H =H ./ H(3,3);
  
end

