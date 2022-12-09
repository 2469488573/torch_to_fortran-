        program main 
 
        use bridge , only: tbf
         
        implicit none 
        integer              :: m = 4
        real,allocatable     :: x_array(:,:)
        real                 :: y_cesm(5)
        allocate(x_array(m,5))
        x_array(:,1) = (/264.32004,0.3210011,14510.625,52310.562/)
        x_array(:,2) = (/264.31717,0.32086015,14449.125,52227.875/)
        x_array(:,3) = (/264.31717,0.32067218,14449.125,52186.5/)
        x_array(:,4) = (/264.31573,0.3205077,14449.125,52062.375/)
        x_array(:,5) = (/264.31573,0.3203667,14387.5,51979.688/)

        call tbf(5,m,x_array,y_cesm)
        print*,y_cesm 
        end program main 
