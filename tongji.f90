        module  tongji


        implicit none
        private
        public mean
       
        contains

        subroutine mean(x,n,x_mean)

                integer  :: n 
                integer  :: i 
                real :: x(n)
                real :: x_mean

                x_mean = 0.0
                do i =1,n
                x_mean = x_mean + x(i)
                end do
                x_mean = x_mean/n
        end subroutine mean

        end module tongji
