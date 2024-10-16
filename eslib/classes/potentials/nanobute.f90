subroutine nanotubecylinder(pot, forces)
    ! Cylinder along x direction that mimics a 6,6 carbon nanotube
    ! Analytical form: V(r)=a2*r^2+a4*r^4+a6*r^6+a8*r^8
    ! and parameters following Dellago and Naor, CPC 169 (2005) 36--39
    implicit none
    real*8 :: a2, a4, a6, a8
    real*8  :: pot, forces(3, n_atoms), r
    integer i_atom, i_dim
    a2=-0.000101790427
    a4=0.0001362104651
    a6=8.1919580588d-06
    a8=3.188645093e-06
    pot=0.d0
    forces=0.d0
    do i_atom=1, n_atoms
       if (species_name(species(i_atom)).eq.'O') then
          call cylindrical_distance(coords(:,i_atom), r)
          pot=pot+a2*(r**2)+a4*(r**4)+a6*(r**6)+a8*(r**8)
!            forces(i_atom, 1)=0.d0
          if (r.gt.0.d0) then
             do i_dim=2, 3
                forces(i_dim,i_atom)=-(2*a2*r+4*a4*(r**3)+6*a6*(r**5)+8*a8*(r**7))*coords(i_dim, i_atom)/r
             enddo
          endif
       endif
    enddo
    end subroutine