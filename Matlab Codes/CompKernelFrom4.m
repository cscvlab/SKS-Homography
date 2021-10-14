%
% 2019-09-04 hammer
% write the function 'CompKernelFrom4' to compute the kernel
% transformation H_K under 4-point case with the first solving process.
% i.e. H_K = H_E*H_T2_inv*H_C*H_T1*H_E. Here H_K is decomposed into five parts, 
% the middle three of which consist of a hyperbolic similarity transformation.
%

%   Detailed explanation goes here
%   Inputs: PQ_src--two source points;      PQ_tar--two target points;   
%           Both of them (3*2 matrices) are in homogeneous coordinates.
%           H_S_src--similar transformation on source plane
%           H_S_tar--similar transformation on target plane
%           If the number of points is greater than 4, please call another funciton
%           'CompKernelFromN' specially designed for optimal estimation under n-point case.
%   Outputs: H_K--kernel transformation between normalized source and target planes;
%   FLOPS: 2*13+2*15+15+8 = 79


function [ H_K ] = CompKernelFrom4( H_S_src, H_S_tar, PQ_src, PQ_tar )
    
    % default elementary transformation
    H_E = [1 0 0; 0 0 1; 0 1 0];  

    % deal with the third point correspondence {P1---P2}
    P1 = PQ_src(:,1);
    P5 = H_E*H_S_src*P1;
    P2 = PQ_tar(:,1);
    P6 = H_E*H_S_tar*P2;    
    P5 = P5./P5(3);
    P6 = P6./P6(3);
    % compute the coordinates of P5 or P6,  actually 13 flops, e.g.,
	%									        [ 1       ]   [ a1  -b1    ]   [ 1     -O1.x  ]   [ P1.x ]     [ P5.x ]
	% P5 = H_E * H_S_src * P1 = [       1 ] * [ b1   a1    ] * [    1  -O1.y ] * [ P1.y ] = [ P5.y ]
	%									        [    1    ]   [		         1]    [            1    ]     [  1   ]     [   1  ]
    H_T1 = [1 0 -P5(1); 0 1 -P5(2); 0 0 1];         % translation transformation
    H_T2 = [1 0 -P6(1); 0 1 -P6(2); 0 0 1];
    H_T2_inv = [1 0 P6(1); 0 1 P6(2); 0 0 1];      % inverse matrix

    % deal with the fourth point correspondence {Q1---Q2}
    Q1 = PQ_src(:,2);
    Q7 = H_T1*H_E*H_S_src*Q1;    
    Q2 = PQ_tar(:,2);
    Q8 = H_T2*H_E*H_S_tar*Q2;    
    x7 = Q7(1)/Q7(3);
    y7 = Q7(2)/Q7(3);    
    x8 = Q8(1)/Q8(3);
    y8 = Q8(2)/Q8(3);   
    % compute the coordinates of Q7 or Q8,  actually 15 flops, e.g.,
	%									                   [ 1       -P5.x ]    [ 1       ]   [ a1  -b1    ]   [ 1     -O1.x  ]   [ Q1.x ]     [ Q7.x ]
	% Q7 = H_T1 * H_E * H_S_src * Q1 = [     1   -P5.y ] * [       1 ] * [ b1   a1    ] * [    1  -O1.y ] * [ Q1.y ] = [ Q7.y ]
	%									                   [              1   ]    [    1    ]   [		         1]    [            1    ]     [  1   ]     [   1  ]
    a_C = (y8*y7-x8*x7) / (y7*y7-x7*x7);         % solve linear equations, 15 flops
    b_C = (x8*y7-y8*x7) / (y7*y7-x7*x7);

    % compute kernel transformation H_K by matrices multiplication
    H_C = [a_C b_C 0 ; b_C a_C 0 ; 0 0 1];        % hyperbolic scale and rotation transformation
    H_SH = H_T2_inv*H_C*H_T1;                     %  hyperbolic similarity transformation
    H_K = H_E*H_SH*H_E;     % kernel transformation
    % actually 8 flops
	%			   [ 1         ]   [ 1       P6.x ]    [ a_C   b_C    ]   [ 1       -P5.x ]    [ 1         ]
	%	H_K = [         1 ] * [     1   P6.y ] * [ b_C   a_C    ] * [     1   -P5.y ] * [         1 ]
	%			  [     1     ]    [             1  ]    [                  1]   [               1  ]    [     1     ]
	%				    [ a_C   P6.x-a_C*P5.x-b_C*P5.y   b_C ]
	%				= [ 0               1               0 ]
	%				    [ b_C   P6.y-a_C*P5.y-b_C*P5.x   a_C ]

end

