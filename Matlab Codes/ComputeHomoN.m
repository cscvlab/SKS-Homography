% 2019-09-04 hammer
% write the function 'ComputeHomoN' to compute homography with N point corresdences
% 
% the part of Gauss-Newton code refers to 'compute_homography.m' in
% Calibration Toolbox implemented by Jean-Yves Bouguet.

%   Detailed explanation goes here
%   INputs: SourcePts---source points;   TargetPts---target points;
%               indVec---inputting indices if nesscessary
%           Both of them (3*N matrices) are in homogeneous coordinates.
%           If the number of points is equal to 4, please call another funciton
%           'ComputeHomo4' specially designed for 4-point case.
%   Outputs: H---linear homography between source and target;
%                  H_opt---optimal homography by 1-step Gauss-Newton


function [ H, H_opt ] = ComputeHomoN( SourcePts, TargetPts,  indVec )
    
    iteration = 1;

    Np = length(SourcePts(1,:));     % N
    
    % determine TAP
    if (nargin<3)   % if input terms is less than 3, open the TAP selection
        % select TAP with the longest distance from sequential sRatio*N times sampling
        sRatio = 0.2;                  % sampling ratio
        %sRatio = 0.5;                  % for robust test
        interval = floor(1/sRatio);
        if Np < 10                     % if points number <10, increase the number of samples
            interval = interval/2;
        end
        seq_ind = [1:interval:Np];
        %seq_ind = [1:interval:Np, Np];
        max_len = 0;              
        max_ind1 = seq_ind(1);            % index of TAP with the longest distance
        max_ind2 = seq_ind(2); 
        for i=1:length(seq_ind)-1
            len = abs( SourcePts(1,seq_ind(i)) - SourcePts(1,seq_ind(i+1)) ) ...
                + abs( SourcePts(2,seq_ind(i)) - SourcePts(2,seq_ind(i+1)) );       % L1 distance
            if len > max_len
                max_len = len;
                max_ind1 = seq_ind(i);
                max_ind2 = seq_ind(i+1);
            end
        end
    else % if input terms is equal to 3, directly use the inputting indices (e.g., for chessboard pattern)
        max_ind1 = indVec(1);
        max_ind2 = indVec(2);
    end

    % compute H_S1 on source plane by TAP 
    M1 = SourcePts(:,max_ind1);
    N1 = SourcePts(:,max_ind2);
    O1 = 0.5*(M1+N1);      % midpoint
    ON1 = N1 - O1;            % vector ON1
    k1 = 1/sum(ON1.^2);
    a1 = k1*ON1(1);
    b1 = -k1*ON1(2);
    H_T_1 = [1 0 -O1(1); 0 1 -O1(2); 0 0 1];      % translation matrix in H_S1 
    H_AB_1 = [a1 -b1 0; b1 a1 0; 0 0 1];           % scale and rotation matrix in H_S1 
    H_S1 = H_AB_1*H_T_1;                                % similar transformation on source plane
    
    % compute H_S2 on target plane by TAP 
    M2 = TargetPts(:,max_ind1);
    N2 = TargetPts(:,max_ind2);
    O2 = 0.5*(M2+N2);      % midpoint
    ON2 = N2 - O2;            % vector ON1
    k2 = 1/sum(ON2.^2);
    a2 = k2*ON2(1);
    b2 = -k2*ON2(2);
    H_T_2 = [1 0 -O2(1); 0 1 -O2(2); 0 0 1];     
    H_AB_2 = [a2 -b2 0; b2 a2 0; 0 0 1];
    H_S2 = H_AB_2*H_T_2;                               % similar transformation on target plane
    H_S2_inv = [ON2(1) -ON2(2) O2(1); ON2(2) ON2(1) O2(2); 0 0 1];      % inverse matrix of H_S2
    
    % utilize N-2 pairs of points to linearly estimate kernel transformation H_K
    X1 = SourcePts;
    X1(:,[max_ind1 max_ind2]) = [];
    X2 = TargetPts;
    X2(:,[max_ind1 max_ind2]) = [];
    X3 = H_S1*X1;       % (N-2)*8 flops
    X4 = H_S2*X2;       % (N-2)*8 flops
    H_K = CompKernelFromN( X3, X4 );          % H_K---kernel transformation
    H = H_S2_inv*H_K*H_S1;
    H = H./H(3,3);                                            % H---linear solution of homography
    
    %% 1-step Gauss-Newton iteration optimalization of H
    X4 = H_S2*TargetPts;
    X4_repro = H_K*H_S1*SourcePts;
    X4_repro = X4_repro./ (ones(3,1)*X4_repro(3,:));
    AA = eye(3);
    hhv = reshape(AA',9,1);
    hhv = hhv(1:8);
    
    % test---observe the change of reprojection errors
    TarPts_est = AA*X4_repro;                  % reprojection points
    err = TarPts_est(1:2,:) - X4(1:2,:);         % reprojection error
    sum1 = 0;
    for ii=1:Np
        c=sqrt(err(1,ii)^2 + err(2,ii)^2);
        sum1 = sum1 + c;
    end
    err_old = sum1/Np;
    
    % Gauss-Newton iteration
    for iter=1:iteration
         mrep = AA * X4_repro;
         J = zeros(2*Np,8);
         MMM = (X4_repro ./ (ones(3,1)*mrep(3,:)));
         J(1:2:2*Np,1:3) = -MMM';
         J(2:2:2*Np,4:6) = -MMM';
         mrep = mrep ./ (ones(3,1)*mrep(3,:));
         m_err = X4(1:2,:) - mrep(1:2,:);
         m_err = m_err(:);
         MMM2 = (ones(3,1)*mrep(1,:)) .* MMM;
         MMM3 = (ones(3,1)*mrep(2,:)) .* MMM;
         J(1:2:2*Np,7:8) = MMM2(1:2,:)';
         J(2:2:2*Np,7:8) = MMM3(1:2,:)';
         % MMM = (X4_repro ./ (ones(3,1)*mrep(3,:)))';
         hh_innov  = inv(J'*J)*J'*m_err;
         hhv_up = hhv - hh_innov;
         H_up = reshape([hhv_up;1],3,3)';
         hhv = hhv_up;
         AA = H_up;

         % test
         TarPts_est = AA*X4_repro;            % reprojection points
         err = TarPts_est(1:2,:) - X4(1:2,:);   % reprojection error
         sum1 = 0;
         for ii=1:Np
             c=sqrt(err(1,ii)^2 + err(2,ii)^2);
             sum1 = sum1 + c;
         end
         err_new = sum1/Np;
         err_up = err_old-err_new;
         err_old = err_new;
    end
    
    % final optimal result
    H_opt = H_S2_inv*AA*H_K*H_S1;
    H_opt = H_opt./H_opt(3,3);

end

