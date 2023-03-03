%  Affine-Core-Affine (ACA) Decomposition of 2D homographies for a rectangle on source plane
%
%              H = H_A2^(-1) * H_C * H_A1
%
%  FLOPs: 59 = 47 (compute homographies up to a scale from 4 point correspondences) + 12 (normalization)

%  Detailed explanation goes here
%  Inputs: TargetPts --- target points;   
%              M_x, M_y --- 2D coordinates of upper-left vertex of source rectangle; 
%              width, ratio_rec --- width and ratio aspect of source rectangle;
%  Outputs: H --- homography between source and target planes;

%  License
%
%


function [ H ] = ACA_rect ( TargetPts,  M_x, M_y, width, ratio_rec )

    % compute required vectors
    sub_tar = TargetPts(1:2,2:4) - TargetPts(1:2,1);        % 2 vector operations (VO) or 6 flops 
    c = cross( sub_tar(2,:), sub_tar(1,:) );    % 1 VO or 9 flops,   c = [Q4(1) Q4(2) -f_A2];
    b= sum(c) * TargetPts(:,1);    %  1 VO or 4 flops (for C procedure),  sum(c) = -t_2
    % compute homography up to a scale
    H(:,1) = TargetPts(:,2)*c(1) - b;       %  2 VO or 5 flops
    H(:,2) = ratio_rec * (TargetPts(:,3)*c(2) - b);       %  3 VO or 8 flops. If a square is given (ratio_rec=1), only 2 VO or 5 flops is required.
    H(:,3) = width*b - M_x*H(:,1) - M_y*H(:,2);       %  5 VO or 15 flops
    
    % normalization based on the last element of H, 12 flops
    % in deep homography pipeline, this step can be removed as the
    % possible subsequent image warping does not require normalization.
    H =H ./ H(3,3);
    
end

