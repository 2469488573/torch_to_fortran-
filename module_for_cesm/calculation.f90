        module  calculation 
        !this module is for calculation 

        implicit none 
        private 
        public cal_output
        public cal_input
        public cal_dense
        public cal_jihuo
        contains 

        subroutine cal_input(x,w,b,y,m,n)
        !计算输入层
                ! input variable
                integer,intent(in)    ::  m,n
                
                real,intent(in)       ::  x(m),b(n),w(n,m)

                !local variable
                integer               :: i,j,k 
                real                  ::wx(n,m),wx_sum(n),&
                                        wxb(n)

                !output variable
                real,intent(out)      :: y(n)

                wx = 0
                wx_sum =0 
                wxb = 0
                
                do i = 1,n!对节点循环
                  
                  do j = 1,m!对因子循环
                        wx(i,j) = w(i,j) * x(j)
                  end do
                  
                  do j =1,m!对因子循环
                        wx_sum(i) = wx_sum(i) + wx(i,j)
                  end do
                  
                  wxb(i) = wx_sum(i) + b(i)

                enddo

                y = wxb

        end subroutine cal_input


        subroutine cal_jihuo(x,n,function_kind,y)
!这个函数用来计算激活函数relu(x),tanh(x),sigmod(x)
                integer,intent(in) :: n,function_kind
                real,intent(in)    :: x(n)
                real,intent(out)   :: y(n) 
                integer            :: i
!               select
                do i = 1,n
                  if (x(i)>0)then
                        y(i) = x(i)
                  else
                        y(i) = 0
                  end if
                enddo
                  !relu_wxb(i) = max(0,wxb(i))

        end subroutine cal_jihuo 

        subroutine cal_dense(x,w,b,y,n,m)
!这个函数计算中间层
                ! input variable
                integer,intent(in)    ::  m,n
                
                real,intent(in)       ::  x(m),b(n),w(n,m)

                !local variable
                integer               :: i,j,k 
                real                  ::wx(n,m),wx_sum(n),&
                                        wxb(n)

                !output variable
                real,intent(out)      :: y(n)

                wx = 0
                wx_sum =0 
                wxb = 0
                
                do i = 1,n!对节点循环
                  
                  do j = 1,m!对因子循环
                        wx(i,j) = w(i,j) * x(j)
                  end do
                  
                  do j =1,m!对因子循环
                        wx_sum(i) = wx_sum(i) + wx(i,j)
                  end do
                  
                  wxb(i) = wx_sum(i) + b(i)

                enddo

                y = wxb


        end subroutine cal_dense


        subroutine cal_output(x,c,d,n,y)
        ! this code is for calculate y = c*x + d
                !input variable
                integer,intent(in)      :: n
                real,intent(in)         :: x(n),c(n)
                real,intent(in)         :: d

                !local variable 
                integer                 :: i 
                
                !output variable 
                real,intent(out)        :: y
                
                y = 0

                do i = 1,n

                        y = y + c(i) * x(i)

                end do
                
                y = y + d
                !输出结果是一个实数 

        end subroutine cal_output

       end module calculation       
