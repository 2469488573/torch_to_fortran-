      module   bridge 


        !|===============================================================|!
        !|                                                               |!
        !|                     torch-bridge-fortran                      |！
        !|                                                               |！
        !|===============================================================|!
        !|本程序为从pyTorch 到 Fortran 的 一个接口，致力于将现代算法     |!
        !|训练的模型加到使用fortran编写的大气海洋等大型模式中。          |!
        !|作者：马钰斌                                                   |!
        !|开始日期：2022年11月27日                                       |!
        !|更新记录：                                                     |!
        !|===============================================================|!

                                                                      
!______________________________________________________________________ 
!                    ||                         ||                     !          
!                   /||\                       /||\                    !        
!         Torch    / || \         Bridge      / || \   Fortran         !                  
!                 / /||\ \                   / /||\ \                  !                            
!                / / || \ \                 / / || \ \                 !           
!        _______/_/_/||\_\_\_______________/_/_/||\_\_\_______         !
!        ============||=========================||=============        !
!          ~~~~~~    ||       ~~~~~~~~~~        ||  ~~~~~~             ! 
!         ~~~~~~~~~  ||      ~~~~~~~~~~~~~      || ~~~~~~~~~           !       
!         ~~~~~~~~~~~||    ~~~~~~~~~~~~~~~~~    || ~~~~~~~~~~~~        !    
!         ~~~~~~~    ||  ~~~~~~~~~~~~~~~~~~~~~~ || ~~~~~~~             !   
!                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                  !                 
!______________________________________________________________________!                                                                                

        implicit none 
        private
        public tbf


       contains

        subroutine logo()

        print*,'             ||                         ||             '
        print*,'            /||\                       /||\            '
        print*,'  Torch    / || \         Bridge      / || \   Fortran '
        print*,'          / /||\ \                   / /||\ \          '
        print*,'         / / || \ \                 / / || \ \         '
        print*,' _______/_/_/||\_\_\_______________/_/_/||\_\_\_______ '
        print*,' ============||=========================||============ '
        print*,'   ~~~~~~    ||       ~~~~~~~~~~        ||  ~~~~~~     '
        print*,'  ~~~~~~~~~  ||      ~~~~~~~~~~~~~      || ~~~~~~~~~   '
        print*,'  ~~~~~~~~~~~||    ~~~~~~~~~~~~~~~~~    || ~~~~~~~~~~~ '
        print*,'  ~~~~~~~    ||  ~~~~~~~~~~~~~~~~~~~~~~ || ~~~~~~~     '
        print*,'          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~          '
        print*,'______________________________________________________ '

        end subroutine logo

        subroutine tbf(dirname,ncol,m_cesm ,x_cesm,y_cesm )
!======================================================================
        !主程序引用mod区域

        !深度学习等需要用到统计方法，相关的代码写在tongji中        
        use tongji,     only:mean
        !文件读写相关的代码
        use file_io,    only:danzhi,shuzu,filelinenum,weiducanshu,&
                             array_1d,array_2d,array_b_dense,array_w_dense
        !深度学习计算的相关代码
        use calculation,only:cal_input,cal_jihuo,cal_dense,cal_output
        ! 主程序定义变量区域
        character(len = 100) :: dirname       !深度学习模型所在路径
        integer              :: m,n,o            !w的维度，分别表示因子数，每层节点数，层数
        integer              :: o_c              !当前计算的层数
        integer              :: i,j,k            !索引
        integer              :: error            !allocate flag
        integer              :: function_kind = 1!激活函数的种类
        !计算中间变量都定义为可变数组，之后根据输入来分配内存空间
        real,allocatable     ::w(:,:,:),x(:),b(:,:),c(:),y(:,:),&
                               w_input(:,:)
        real                 :: d,z
        !文件名字符数组都定义长度为100，文件名不宜过长       
!       character(len = 100) :: dirname ="/data/chengxl/&
!                         pblh_deeplearning/torch_bridge_fortran/python/"
        character(len = 100) ::  filename_canshu  ,& 
                                 filename_w1      ,& 
                                 filename_b1      ,& 
                                 filename_w       ,& 
                                 filename_b       ,& 
                                 filename_c       ,& 
                                 filename_d      



 
        real,allocatable     :: x_array(:,:)
!从cesm中传进来数据
        integer,intent(in)   :: ncol,m_cesm 
        real,intent(in)      :: x_cesm(m_cesm,ncol)
        real,intent(out)     :: y_cesm(ncol)
!===================================================================================================


        call logo()


        print*,'                     程序开始运行                      '
        print*,'======================================================='
!给定文件位置                                       
                filename_canshu =  trim(dirname)//'shuchucanshu.txt'  
                filename_w1     =  trim(dirname)//'w_input.txt'   
                filename_b1     =  trim(dirname)//'b_input.txt'    
                filename_w      =  trim(dirname)//'w_dense.txt'    
                filename_b      =  trim(dirname)//'b_dense.txt'     
                filename_c      =  trim(dirname)//'w_output.txt'   
                filename_d      =  trim(dirname)//'b_output.txt'

!因子数，节点数，神经层数初始化，默认为三个变量，四个节点和两层神经网络
        m=3
        n=5
        o=2

!将pytorch的参数传进来,m,n,o

        call weiducanshu(filename_canshu,n,m,o)

!给可变数组分配内存

        allocate(w(n,m,o),stat = error)
        allocate(x(m),stat = error)

        allocate(b(n,o),stat = error)

        allocate(c(n),stat = error)

        allocate(y(n,o+1),stat = error)

        allocate( w_input(n,m),stat = error)


!最初赋值，防止内存错误,先用隐式构造一维数组，然后再改成多维数组

        w=reshape( (/(i*0+1,i=1,n*n*o,1)/),(/n,n,o/))

        x=(/(i*0+1,i = 1,m,1)/)

        b=reshape((/(i*0+1, i=1,n*o,1)/),(/n,o/))

        c=(/(i*0+1 ,i = 1,n,1) /)        

        d=1

        y=reshape((/(i*0,i = 1,n*(o+1),1)/) ,(/n,o+1/))

        w_input = reshape((/(i*0+1,i = 1,n*m,1 )/),  (/n,m/) )

!检查变量是否初始化正确
!        print*,'x = ', x
!        print*, 'b = ',b
!        print*,'c = ',c
!        print*,'w = ',w

!=========================================================================
!输入从pytorch中传进来的参数！    w b c d 

        !输入w_input
        call array_2d(filename_w1,n,m,w_input)
        call array_2d(filename_w1,n,m,w_input)
        !输入b_input
        call array_1d(filename_b1,n,b(:,1))
        call array_1d(filename_b1,n,b(:,1))
!为了测试多次读写运行两次

     !----------------------------------------------------------------- 

        !输入w_dense (存在问题)
        call array_w_dense(filename_w,n,n,o-1,w(:,:,2:) )

        !输入b_dense
        call array_b_dense(filename_b,n,o-1,b(:,2:))

     !----------------------------------------------------------------- 

        !输入w_output 或者叫c
        call array_1d(filename_c,n,c)

        !输入b_output 或者叫d
        call danzhi(filename_d,d)

!模型参数输出结束。        
!=========================================================================

!对x赋值

         x=(/264.32004,6417.3438,0.3210011/)

        allocate(x_array(m,5))
        x_array(:,1) = (/264.32004,6417.3438,0.3210011,271.46045,263.9699,14510.625,-33286.188,101503.36,52310.562/)
        x_array(:,2) = (/264.31717,6399.625,0.32086015,271.46045,263.9699,14449.125,-33387.406,101504.21,52227.875/)
        x_array(:,3) = (/264.31717,6399.625,0.32067218,271.46045,263.9699,14449.125,-33488.656,101504.21,52186.5/)
        x_array(:,4) = (/264.31573,6390.75,0.3205077,271.46045,263.9699,14449.125,-33589.875,101505.92,52062.375/)
        x_array(:,5) = (/264.31573,6390.75,0.3203667,271.46045,263.97174,14387.5,-33691.125,101505.92,51979.688/)



!========================================================================
!新建立一个变量叫x_array，然后对x_array(:,i)进行循环，然后输出z(i)数组

        do i = 1,ncol



!深度学习模型计算结果代码

! 计算 输入层

        call cal_input(x_cesm(:,i),w_input,b(:,1),y(:,1),m,n)

!        write(*,50),'第1层计算 sum (w1 * x) + b1 = ',y(:,1)

! 计算 激活函数

        !设定激活函数的种类
        !function_kind = 1 !, ReLU
        !function_kind = 2 !, tanh
        !function_kind = 3 !, sigmod

        call cal_jihuo(y(:,1),n,function_kind,y(:,1))

!       write(*,50),'第1层结束 h1 = relu( sum ( w1 * x ) + b1 ) = ',y(:,1)

! 计算 中间层
! w的第一层和第二层维数、是不一样的，引入w1，而不用w(:,:,1),为了w下标和b保持一致


!递推公式:    
!        call cal_dense(y(:,1),w(:,:,2),b(:,2),y(:,2),n,m)
!        print*,'第二层计算 w2 * h1 + b2',y(:,2)
!        call cal_jihuo(y(:,2),n,function_kind,y(:,2))
!        print*,'第二层结束 h2 = relu( w2 * h1 + b2 )',y(:,2)
!        call cal_dense(y(:,2),w(:,:,3),b(:,3),y(:,3),n,m)
!        print*,'第',3,'层计算 w3 * h2 + b3',y(:,3)
!        call cal_jihuo(y(:,3),n,function_kind,y(:,3))
!        print*,'第',3,'层结束 h3 = relu( w3 h2 + b3 )',y(:,3)
! 激活函数时不变y的下标，计算线性函数时改变下标，x总是小于w,b,y

! 对中间层循环优化

        do k = 2,o

        call cal_dense(y(:,k-1),w(:,:,k),b(:,k),y(:,k),n,n)
!        write(*,100)'第',k,'层计算 sum ( w',k,' * h',k-1,') + b',k,' = ',y(:,k)

        call cal_jihuo(y(:,k),n,function_kind,y(:,k))
!        write(*,150),'第',k,'层结束 h',k,' = relu(sum( w',k,' * h',k-1,') + b',k,' ) = ',y(:,k)

        enddo

! 计算 输出层

        call cal_output(y(:,o),c,d,n,y_cesm(i))
        write(*,200) '输出结果 z = ',y_cesm(i)

        enddo


!格式化输出，输出格式定义
50      format('',A,/(F10.2))
100     format('',A,I0,A,I0,A,I0,A,I0,A,/(F10.2))
150     format('',5(A,I0),A,/(F10.2))
200     format('',A,/F12.6)

        print*,'                     程序运行结束                      '
        print*,'======================================================='


!        print*,"子程序运行结束"
!=========================================================================
        end subroutine tbf

      end module bridge
