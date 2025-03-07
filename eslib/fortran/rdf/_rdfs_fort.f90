!**********************************************************!
!
! THIS FILE HAS BEEN ADAPTED FROM I-PI (i-pi/tools/f90)
!
!**********************************************************!

SUBROUTINE InteratomicDistances(positions, cell, invCell, nsnapshots, natoms, distances)
  IMPLICIT NONE

  ! Input
  INTEGER, INTENT(in) :: nsnapshots, natoms
  REAL(8), INTENT(in) :: positions(nsnapshots, 3*natoms), cell(3,3), invCell(3,3)

  ! Output
  REAL(8), INTENT(out) :: distances(nsnapshots*natoms, nsnapshots*natoms)

  ! Local variables
  INTEGER :: n, m, i, j
  REAL(8) :: dAB, vdAB(3)

  ! Initialize distances array
  distances(:,:) = 0.0

  ! Loop over snapshots and atoms
  DO n = 1, nsnapshots
    DO i = 1, natoms
      DO m = n, nsnapshots
        DO j = 1, natoms
          CALL CalcMinDist(cell, invCell, &
                           positions(n, 3*i-2), positions(n, 3*i-1), positions(n, 3*i), &
                           positions(m, 3*j-2), positions(m, 3*j-1), positions(m, 3*j), &
                           dAB, vdAB)
          distances((n-1)*natoms+i, (m-1)*natoms+j) = dAB
          distances((m-1)*natoms+j, (n-1)*natoms+i) = dAB
        END DO
      END DO
    END DO
  END DO

END SUBROUTINE InteratomicDistances

subroutine InterMolecularRDF(gOOr, atxyz1, atxyz2, r_min, r_max, &
    cell, invCell, partition1, partition2, mass1, mass2, &
    nsnapshots, nat1, nat2, nbins)
  !
  IMPLICIT NONE
  !
  ! Input
  !
  INTEGER, INTENT(in) :: nsnapshots, nat1, nat2, nbins
  REAL(8), INTENT(in) :: mass1, mass2
  REAL(8), INTENT(in) :: r_min, r_max
  REAL(8), INTENT(in) :: atxyz1(nsnapshots,nat1,3), atxyz2(nsnapshots,nat2,3), cell(3,3), invCell(3,3)
  INTEGER, INTENT(in) :: partition1(nsnapshots,nat1), partition2(nsnapshots,nat2)
  !
  ! Output
  !
  REAL(8), INTENT(inout) :: gOOr(nbins,2)
  !
  ! Local variables
  !
  INTEGER :: ia, ib, ig, n
  REAL(8) :: deltar, dAB, vdAB(3), norm
  REAL(8), PARAMETER :: tol = 0.00001
  !
  ! Histogram step initialization
  !
  deltar = gOOr(2,1) - gOOr(1,1)
  !
  ! Start computing g(r) from MD configurations
  !
  ! Normalization constant
  !
  IF (mass1.EQ.mass2) THEN
    norm = 1.0/(nat1*(nat2-1))
  ELSE
    norm = 1.0/(nat1*nat2)
  END IF
  !
  ! Populate histogram bins for gOO(r)...
  !
  DO n=1,nsnapshots! Loop over snapshots
    DO ia=1,nat1
      DO ib=1,nat2
        !
        IF (partition1(n,ia) .NE. partition2(n,ib)) THEN ! different molecule: compute
          !
          ! Compute the distance of the closest image of atom B to atom A using minimum image convention...
          CALL CalcMinDist(cell, invCell, &
            atxyz1(n,ia,1), atxyz1(n,ia,2), atxyz1(n,ia,3), &
            atxyz2(n,ib,1), atxyz2(n,ib,2), atxyz2(n,ib,3), &
            dAB, vdAB)
          ! Screen distances that are outside desired range
          IF (dAB.LT.r_max.AND.dAB.GT.r_min) THEN
            ig=INT((dAB-r_min)/deltar)+1  !bin/histogram position
            gOOr(ig,2)=gOOr(ig,2)+1*norm
          END IF
          !
        END IF
      END DO !ib 
    END DO !ia 
  END DO !n
  !
END SUBROUTINE InterMolecularRDF

subroutine UpdateQRDFFixedCell(gOOr, atxyz1, atxyz2, r_min, r_max, cell, invCell, mass1, mass2, nsnapshots, nat1, nat2, nbins)
  !
  IMPLICIT NONE
  !
  ! Input
  !
  INTEGER, INTENT(in) :: nsnapshots, nat1, nat2, nbins
  REAL(8), INTENT(in) :: mass1, mass2
  REAL(8), INTENT(in) :: r_min, r_max
  REAL(8), INTENT(in) :: atxyz1(nsnapshots,3*nat1), atxyz2(nsnapshots,3*nat2), cell(3,3), invCell(3,3)
  !
  ! Output
  !
  REAL(8), INTENT(inout) :: gOOr(nbins,2)
  !f2py intent(hide) :: nat1
  !f2py intent(hide) :: nat2
  !f2py intent(hide) :: nbins
  ! Local variables
  !
  INTEGER :: ia, ib, ig, n
  REAL(8) :: deltar, dAB, vdAB(3), norm
  REAL(8), PARAMETER :: tol = 0.00001
  !
  ! Histogram step initialization
  !
  deltar = gOOr(2,1) - gOOr(1,1)
  !
  ! Start computing g(r) from MD configurations
  !
  ! Normalization constant
  !
  IF (mass1.EQ.mass2) THEN
    norm = 1.0/(nat1*(nat2-1))
  ELSE
    norm = 1.0/(nat1*nat2)
  END IF
  !
  ! Populate histogram bins for gOO(r)...
  !
  DO n=1,nsnapshots! Loop over snapshots
    DO ia=1,nat1
      DO ib=1,nat2
        ! Compute the distance of the closest image of atom B to atom A using minimum image convention...
        CALL CalcMinDist(cell, invCell, &
          atxyz1(n,3*ia-2), atxyz1(n,3*ia-1), atxyz1(n,3*ia), &
          atxyz2(n,3*ib-2), atxyz2(n,3*ib-1), atxyz2(n,3*ib), &
          dAB, vdAB)
        ! Screen distances that are outside desired range
        IF (dAB.LT.r_max.AND.dAB.GT.r_min) THEN
          ig=INT((dAB-r_min)/deltar)+1  !bin/histogram position
          gOOr(ig,2)=gOOr(ig,2)+1*norm
        END IF
      END DO !ib 
    END DO !ia 
  END DO !n
  !
END SUBROUTINE UpdateQRDFFixedCell

subroutine UpdateQRDFVariableCell(gOOr, atxyz1, atxyz2, r_min, r_max, cell, invCell, mass1, mass2, nat1, nat2, nbins)
    !
    IMPLICIT NONE
    !
    ! Input
    !
    INTEGER, INTENT(in) :: nat1, nat2, nbins
    REAL(8), INTENT(in) :: mass1, mass2
    REAL(8), INTENT(in) :: r_min, r_max
    REAL(8), INTENT(in) :: atxyz1(3*nat1), atxyz2(3*nat2), cell(3,3), invCell(3,3)
    !
    ! Output
    !
    REAL(8), INTENT(inout) :: gOOr(nbins,2)
    !f2py intent(hide) :: nat1
    !f2py intent(hide) :: nat2
    !f2py intent(hide) :: nbins
    ! Local variables
    !
    INTEGER :: ia, ib, ig
    REAL(8) :: deltar, dAB, vdAB(3), norm
    REAL(8), PARAMETER :: tol = 0.00001
    !
    ! Histogram step initialization
    !
    deltar = gOOr(2,1) - gOOr(1,1)
    !
    ! Start computing g(r) from MD configurations
    !
    ! Normalization constant
    !
    IF (mass1.EQ.mass2) THEN
      norm = 1.0/(nat1*(nat2-1))
    ELSE
      norm = 1.0/(nat1*nat2)
    END IF
    !
    ! Populate histogram bins for gOO(r)...
    !
    DO ia=1,nat1
      DO ib=1,nat2
        ! Compute the distance of the closest image of atom B to atom A using minimum image convention...
        CALL CalcMinDist(cell, invCell, &
          atxyz1(3*ia-2), atxyz1(3*ia-1), atxyz1(3*ia), &
          atxyz2(3*ib-2), atxyz2(3*ib-1), atxyz2(3*ib), &
          dAB, vdAB)
        ! Screen distances that are outside desired range
        IF (dAB.LT.r_max.AND.dAB.GT.r_min) THEN
          ig=INT((dAB-r_min)/deltar)+1  !bin/histogram position
          gOOr(ig,2)=gOOr(ig,2)+1*norm
        END IF
      END DO !ib 
    END DO !ia 
    !
END SUBROUTINE UpdateQRDFVariableCell

subroutine UpdateQRDFBeadVariableCell(gOOr, atxyz1, atxyz2, r_min, r_max, cell, invCell, mass1, mass2, nbeads, nat1, nat2, nbins)
  !
  IMPLICIT NONE
  !
  ! Input
  !
  INTEGER, INTENT(in) :: nbeads, nat1, nat2, nbins
  REAL(8), INTENT(in) :: mass1, mass2
  REAL(8), INTENT(in) :: r_min, r_max
  REAL(8), INTENT(in) :: atxyz1(nbeads,3*nat1), atxyz2(nbeads,3*nat2), cell(3,3), invCell(3,3)
  !
  ! Output
  !
  REAL(8), INTENT(inout) :: gOOr(nbins,2)
  !f2py intent(hide) :: nbeads
  !f2py intent(hide) :: nat1
  !f2py intent(hide) :: nat2
  !f2py intent(hide) :: nbins
  ! Local variables
  !
  INTEGER :: ia, ib, ig, ih
  REAL(8) :: deltar, dAB, vdAB(3), norm
  REAL(8), PARAMETER :: tol = 0.00001
  !
  ! Histogram step initialization
  !
  deltar = gOOr(2,1) - gOOr(1,1)
  !
  ! Start computing g(r) from MD configurations
  !
  ! Normalization constant
  !
  IF (mass1.EQ.mass2) THEN
    norm = 1.0/(nat1*(nat2-1))
  ELSE
    norm = 1.0/(nat1*nat2)
  END IF
  !
  ! Populate histogram bins for gOO(r)...
  !
  DO ih=1,nbeads
    DO ia=1,nat1
      DO ib=1,nat2
        ! Compute the distance of the closest image of atom B to atom A using minimum image convention...
        CALL CalcMinDist(cell, invCell, &
          atxyz1(ih,3*ia-2), atxyz1(ih,3*ia-1), atxyz1(ih,3*ia), &
          atxyz2(ih,3*ib-2), atxyz2(ih,3*ib-1), atxyz2(ih,3*ib), &
          dAB, vdAB)
        ! Screen distances that are outside desired range
        IF (dAB.LT.r_max.AND.dAB.GT.r_min) THEN
          ig=INT((dAB-r_min)/deltar)+1  !bin/histogram position
          gOOr(ig,2)=gOOr(ig,2)+1*norm
        END IF
      END DO !ib 
    END DO !ia 
  END DO !ih 
  !
END SUBROUTINE UpdateQRDFBeadVariableCell
    
SUBROUTINE CalcMinDist(cell,invCell,xA,yA,zA,xB,yB,zB,dAB,rAB)
    !
    IMPLICIT NONE
    !
    REAL(8), INTENT(IN) :: xA, yA, zA, xB, yB, zB, invCell(3,3), cell(3,3)
    REAL(8), INTENT(OUT) :: rAB(3), dAB
    ! Local
    REAL(8) :: rAB2(3)

    !Initialization of distance
    dAB=0.0
    !
    ! Compute distance between atom A and atom B (according to the minimum
    ! image convention)...
    !
    rAB(1)=xA-xB   ! r_AB = r_A - r_B
    rAB(2)=yA-yB   ! r_AB = r_A - r_B
    rAB(3)=zA-zB   ! r_AB = r_A - r_B
    !
    rAB2(1)=invCell(1,1)*rAB(1)+invCell(1,2)*rAB(2)+invCell(1,3)*rAB(3)   ! s_AB =h^-1 r_AB
    rAB2(2)=invCell(2,1)*rAB(1)+invCell(2,2)*rAB(2)+invCell(2,3)*rAB(3)   ! s_AB =h^-1 r_AB
    rAB2(3)=invCell(3,1)*rAB(1)+invCell(3,2)*rAB(2)+invCell(3,3)*rAB(3)   ! s_AB =h^-1 r_AB
    !
    rAB2(1)=rAB2(1)-IDNINT(rAB2(1))   ! impose MIC on s_AB in range:[-0.5,+0.5]
    rAB2(2)=rAB2(2)-IDNINT(rAB2(2))   ! impose MIC on s_AB in range:[-0.5,+0.5]
    rAB2(3)=rAB2(3)-IDNINT(rAB2(3))   ! impose MIC on s_AB in range:[-0.5,+0.5]
    !
    rAB(1)=cell(1,1)*rAB2(1)+cell(1,2)*rAB2(2)+cell(1,3)*rAB2(3)   ! r_AB = h s_AB(MIC)
    rAB(2)=cell(2,1)*rAB2(1)+cell(2,2)*rAB2(2)+cell(2,3)*rAB2(3)   ! r_AB = h s_AB(MIC)
    rAB(3)=cell(3,1)*rAB2(1)+cell(3,2)*rAB2(2)+cell(3,3)*rAB2(3)   ! r_AB = h s_AB(MIC)
    !
    dAB=DSQRT(rAB(1)*rAB(1)+rAB(2)*rAB(2)+rAB(3)*rAB(3))   ! |r_A -r_B| (MIC)
    !
END SUBROUTINE CalcMinDist
