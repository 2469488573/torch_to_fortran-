        module file_io
! 这个mdule 主要解决从文件中输入输出参数，分为单值和数组

                implicit none 
                private
                public danzhi
                public shuzu
                public filelinenum        
!                public tongyong
                public weiducanshu
        contains

        subroutine weiducanshu(filename,n,m,o)
!从文件中读取m,n,o
                character(len = 100),intent(in) ::filename
                integer,intent(inout)      ::  n,m,o 
                open(unit=13,file=filename)
                read(13,*) 
                read(13,*)
                read(13,*)
                read(13,*), m
                read(13,*)
                read(13,*),n
                read(13,*)
                read(13,*),o
 

        end subroutine weiducanshu

        subroutine danzhi(filename,var)
                !从文件filename中，读取一个数到var中
                character(len = 100),intent(in) ::filename
                real      ::  var 
                open(unit=10,file=filename)
                read(10,100) var
100             format(F8.4) 
                print*,var
        end subroutine danzhi


        subroutine shuzu(filename,hangshu,lieshu,var)
        !从文件中读取多维数组,读取到数组var中
        
                integer,intent(in)  :: hangshu,lieshu
                character(len = 100),intent(in)  :: filename
                integer             :: i,j 
                real                :: var(hangshu,lieshu)
                open(unit =11,file = filename)
                !read(11,100) (var(i,j),i = 1,hangshu)
                do i =1,hangshu
                        do j =1,lieshu
                               read(11,100)  var(i,j)
                        enddo
                enddo
100             format(F8.4)
                print*,'在数组子程序中输出var = ',var


        end subroutine shuzu

        integer function filelinenum(a)
                integer ios
                character a*100
                open(22,file=trim(a))
                filelinenum=0
                do
                        read(22,*,iostat=ios)
                        if(ios/=0)exit
                        filelinenum=filelinenum+1
                end do
                close(22)

        end function filelinenum

!        subroutine tongyong(filename,weidu,var)
!        !读取的txt需要是一行一个数据，先判断行数（数据数），然后再读成一维数组，再根据输入的参数，reshape成多维数组
!        !传进来一个一维数组weidu,这个数组控制reshape函数。
!
!                character(len = 100),intent(in)  :: filename
!                integer,intent(in)               :: weidu_index
!                integer,intent(in)               :: weidu(weidu_index)
!                real,intent(out)                 :: var(n,m,o)
!                real,allocatable                 :: yiweishuzu(:) 
!                integer                          :: length,i
!
!100             format(F10.4)
!
!                length = filelinenum(filename)
!
!                allocate(yiweishuzu(length))
!
!                print*,filelinenum(filename)
!                
!                open(12,file = trim(filename))
!
!                yiweishuzu =0
!
!                do i = 1,length
!                        read(12,100) yiweishuzu(i)                
!                end do
!
!                print*,'yiweishuzu',yiweishuzu
!
!                var =  reshape(yiweishuzu,weidu)
!
!        end subroutine tongyong


      end module file_io
