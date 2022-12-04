      program  bridge 


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

!======================================================================
        !主程序引用mod区域

        !深度学习等需要用到统计方法，相关的代码写在tongji中        
        use tongji,     only:mean
        !文件读写相关的代码
        use file_io,    only:danzhi,shuzu,filelinenum,weiducanshu,&
                             array_1d,array_2d,array_b_dense,array_w_dense
        !深度学习计算的相关代码
        use calculation,only:cal_point,cal_output,cal_input,cal_jihuo,cal_dense

        ! 主程序定义变量区域

        integer              :: m,n,o            !w的维度，分别表示因子数，每层节点数，层数
        integer              :: o_c              !当前计算的层数
        integer              :: i,j,k            !索引
        integer              :: error            !allocate flag
        integer              :: function_kind = 1!激活函数的种类
        !计算中间变量都定义为可变数组，之后根据输入来分配内存空间
        real,allocatable     ::w(:,:,:),x(:),b(:,:),c(:),y(:,:),&
                               w_input(:,:)
        real                 :: d,z
        real                 :: wendu_pingjun
        !文件名字符数组都定义长度为100，文件名不宜过长        
        character(len = 100) :: filename_canshu = 'shuchucanshu.txt',&
                                filename_w1= 'w_input.txt',&
                                filename_b1= 'b_input.txt',&
                                filename_w = 'w_dense.txt',&
                                filename_b = 'b_dense.txt',&
                                filename_c = 'w_output.txt',&
                                filename_d = 'b_output.txt'
        real,allocatable     :: x_array(:,:)
        call logo()


        print*,'                     程序开始运行                      '
        print*,'======================================================='

!因子数，节点数，神经层数初始化，默认为三个变量，四个节点和两层神经网络
        m=3
        n=5
        o=2

!将pytorch的参数传进来,m,n,o

        call weiducanshu(filename_canshu,n,m,o)

!给可变数组分配内存

        allocate(w(n,m,o),stat=error)
!        print*,'w allocate flag = ',error!检查是否分配成功，error =0表示成功
        allocate(x(m),stat = error)

        allocate(b(n,o))

        allocate(c(n))

        allocate(y(n,o+1))

        allocate( w_input(n,m),stat=error)

!        print*,'w_input flag = ', error

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
        do i = 1,n
                do j = 1,m
                        print*,i,'节点',j,'因子','w_input',w_input(i,j)
                enddo
        enddo
        !输入b_input
        call array_1d(filename_b1,n,b(:,1))
        print*,'b_input',b

     !----------------------------------------------------------------- 

        !输入w_dense (存在问题)
        call array_w_dense(filename_w,n,n,o-1,w(:,:,2:) )

!        check w
        do k = 1,o
                do i = 1,n
                        do j = 1,n
                                print*,k,'层',i,'节点',j,'因子',w(i,j,k)
                        enddo
                enddo
        enddo

        !输入b_dense
        call array_b_dense(filename_b,n,o-1,b(:,2:))

!       check b
        do k = 1,o
                do i = 1,n
                        print*,k,'层',i,'节点',b(i,k)
                enddo
        enddo

     !----------------------------------------------------------------- 

        !输入w_output 或者叫c
        call array_1d(filename_c,n,c)
        print*,'c',c
        !输入b_output 或者叫d
        call danzhi(filename_d,d)
        print*,'d',d

!模型参数输出结束。        
!=========================================================================

!对x赋值

         x=(/264.32004,6417.3438,0.3210011/)

        allocate(x_array(m,2))
        x_array(:,1) = x
        x_array(:,2) = (/264.31717,6399.625,0.32086015/)
!现在只能手动输入x数组，未来需要实现，x-->z
!函数的功能，这就需要对下面的代码整合。先输入参数构建好模型，然后再输入x，反复调用模型计算。可以看到下面的模型参数已经完全确定，唯一改变的就是x，将x多维数组放进第一个的x中就可以改变输出的z，需要注意的是输入的是一个x向量。
!========================================================================
!新建立一个变量叫x_array，然后对x_array(:,i)进行循环，然后输出z(i)数组

        do i = 1,2



!深度学习模型计算结果代码

! 计算 输入层

        call cal_input(x_array(:,i),w_input,b(:,1),y(:,1),m,n)

        write(*,50),'第1层计算 sum (w1 * x) + b1 = ',y(:,1)

! 计算 激活函数

        !设定激活函数的种类
        !function_kind = 1 !, ReLU
        !function_kind = 2 !, tanh
        !function_kind = 3 !, sigmod

        call cal_jihuo(y(:,1),n,function_kind,y(:,1))

        write(*,50),'第1层结束 h1 = relu( sum ( w1 * x ) + b1 ) = ',y(:,1)

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
        write(*,100)'第',k,'层计算 sum ( w',k,' * h',k-1,') + b',k,' = ',y(:,k)

        call cal_jihuo(y(:,k),n,function_kind,y(:,k))
        write(*,150),'第',k,'层结束 h',k,' = relu(sum( w',k,' * h',k-1,') + b',k,' ) = ',y(:,k)

        enddo

! 计算 输出层

        call cal_output(y(:,o),c,d,n,z)
        write(*,200) '输出结果 z = ',z

        enddo


!格式化输出，输出格式定义
50      format('',A,/(F10.2))
100     format('',A,I0,A,I0,A,I0,A,I0,A,/(F10.2))
150     format('',5(A,I0),A,/(F10.2))
200     format('',A,/F10.4)

        print*,'                     程序运行结束                      '
        print*,'======================================================='

!=========================================================================

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







      end program bridge
