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
        
        use tongji,     only:mean
        use file_io,    only:danzhi,shuzu,filelinenum,tongyong
        use calculation,only:cal_point,cal_output

        ! 主程序定义变量区域

        integer  :: cishu = 10
        real     :: wendu(10),hsb(10),hlb(10)!test data
        integer  :: m,n,o!w的维度，分别表示因子数，每层节点数，层数
        integer  :: o_c !当前计算的层数
        integer  :: i,j,k !索引
        integer  :: error !allocate flug
        real,allocatable     ::w(:,:,:),x(:),b(:,:),c(:),y(:,:)
        real     :: d,z
        real     :: wendu_pingjun,bias02
        real,allocatable     :: weight02(:,:),weight00(:,:),bias00(:,:)
        character(len = 100) :: filename_w ='weight00.txt',&
                                filename_b ='bias00.txt',&
                                filename_c = 'weight02.txt',&
                                filename_d = 'bias02.txt'


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


        print*,'                     程序开始运行                      '
        print*,'======================================================='


        wendu = (/32,23,25,24,29,30,30,30,30,21/)
        hsb = wendu 
        hlb = wendu 

!因子数，节点数，神经层数初始化，默认为三个变量，四个节点和两层神经网络
        m=3
        n=4
        o=2

!给可变数组分配内存
        allocate(w(n,m,o),stat=error)
        print*,'w allocate flug = ',error!检查是否分配成功，error =0表示成功
        allocate(x(m),stat = error)
        allocate(b(n,o))
        allocate(c(n))
        allocate(y(n,o+1))
        
!最初赋值，防止内存错误,先用隐式构造一维数组，然后再改成多维数组
        w=reshape( (/(i*0+1,i=1,m*n*o,1)/),(/n,m,o/))
        x=(/(i*0+1,i = 1,m,1)/)
        b=reshape((/(i*0+1, i=1,n*o,1)/),(/n,o/))
        c=(/(i*0+1 ,i = 1,n,1) /)        
        d=1
        y=reshape((/(i*0,i = 1,n*(o+1),1)/) ,(/n,o+1/))

!检查变量是否初始化正确
!        print*,'x = ', x
!        print*, 'b = ',b
!        print*,'c = ',c
!        print*,'w = ',w

!输入从pytorch中传进来的参数！       
!        allocate(weight02(n,m))
!        allocate(weight00())


 !       print*,'file_d 有',(filelinenum(filename_d)),'行！'
!        call tongyong(filename_w,n,m,o,w)
!        do i  = 1,m
!                do j = 1,n 
!                        do k = 1,o
!                                print*,i,j,k,w(i,j,k)
!                        enddo
!                enddo
!        enddo
!        print*,'tonyong输出的数组w = ',w
!        call danzhi(filename_d,bias02)
!        call shuzu(filename_c,o,n,weight02)
        !call shuzu(len_filename_w,filename_w,n,m,weight00)
        !call shuzu(len_filename_b,filename_b,n,o,bias00)
        !检查输入的数组的正确性：
!        do i =1,n
!                print*, (weight02(i,j),j=1,m)
!        enddo 


        !计算输入层，隐藏层
        y(:,1) = x(:)!将x 作为 y 的第一层 
       print*,y(:,1)
        print*,'calculating relu(sum_wx+b)'
        o_c = 1
        do  while(o_c<o+1)!递归计算y(n+1) = cal_point(y(n))

                call cal_point(y(:,o_c),w,b,c,d,m,n,o,o_c,y(:,o_c+1))

                o_c = o_c + 1! index +1

        end do
!        print*,'计算完隐藏层后的 y = ',y

        !计算输出
        print*,'begin calculating cr+d'

        call cal_output(y(:,o_c),c,d,n,z)
        print*,'z = ',z
      
        print*,'                     程序运行结束                      '
        print*,'======================================================='
        contains
        
      end program bridge
