!***********************************************************************
! Subroutine: shifted_msd
!
! Description:
! This subroutine calculates the mean squared displacement (MSD) 
! over M time shifts for a given set of atomic positions sampled over 
! nsnapshots. The displacement is computed relative to a reference 
! configuration and normalized by the number of valid snapshots used.
!
! Input Arguments:
! - positions(nsnapshots, natoms, 3): Real(8) array containing atomic 
!   positions at each snapshot. The dimensions represent:
!     nsnapshots - Number of snapshots (time steps).
!     natoms     - Number of atoms.
!     3          - Cartesian coordinates (x, y, z).
!
! - nsnapshots: Integer specifying the total number of snapshots.
! - natoms    : Integer specifying the number of atoms.
! - M         : Integer specifying the maximum time shift for MSD 
!               calculation.
! - verbose   : Logical flag; if .true., prints progress messages for
!               debugging and monitoring. Otherwise, runs silently.
!
! Output Argument:
! - delta_squared(M, natoms): Real(8) array where the element 
!   delta_squared(m, j) represents the average squared displacement 
!   of atom j after a time shift of m.
!
! Assumptions:
! - nsnapshots must be at least 2*M for meaningful results.
! - M and natoms must be greater than 0.
!
! Error Handling:
! - Stops execution if any input constraints are violated.
!
!***********************************************************************

subroutine shifted_msd(positions, delta_squared, verbose, nsnapshots, natoms, M)
    implicit none

    ! Input arguments
    integer, intent(in) :: nsnapshots, natoms, M
    logical, intent(in) :: verbose
    real(8), intent(in) :: positions(nsnapshots, natoms, 3)  ! (time_steps, natoms, 3)

    ! Output argument
    real(8), intent(inout) :: delta_squared(M, natoms)  ! (M, natoms)

    ! Local variables
    integer :: snapshot_idx, atom_idx, ref_idx
    real(8), dimension(natoms, 3) :: ref_positions  ! Reference configuration
    real(8) :: displacement_squared  ! Temporary variable for squared displacement
    integer :: valid_snapshots_count  ! Counter for valid snapshots processed
    integer :: progress  ! Progress variables

    !-------------------------------------------------------------------
    ! Input validation
    !-------------------------------------------------------------------
    if (nsnapshots < 2 * M) then
        print *, "Error: nsnapshots must be at least 2*M"
        stop
    end if
    if (M <= 0) then
        print *, "Error: M must be greater than 0"
        stop
    end if
    if (natoms <= 0) then
        print *, "Error: natoms must be greater than 0"
        stop
    end if

    ! if (verbose) print *, "Input validation successful"

    !-------------------------------------------------------------------
    ! Initialization
    !-------------------------------------------------------------------
    delta_squared(:,:) = 0.0  ! Reset the output array
    valid_snapshots_count = 0  ! Initialize counter for valid snapshots
    progress = 0              ! Initialize progress counter

    ! if (verbose) print *, "Initialization complete"

    !-------------------------------------------------------------------
    ! Main computation loop
    !-------------------------------------------------------------------
    do ref_idx = 1, M
        ! Reference configuration at snapshot ref_idx
        ref_positions = positions(ref_idx, :, :)  

        do snapshot_idx = ref_idx, ref_idx + M - 1
            if (snapshot_idx > nsnapshots) exit  ! Prevent out-of-bounds access

            do atom_idx = 1, natoms
                ! Compute squared displacement of atom atom_idx
                displacement_squared = sum((positions(snapshot_idx, atom_idx, :) - ref_positions(atom_idx, :))**2)
                delta_squared(snapshot_idx - ref_idx + 1, atom_idx) = &
                    delta_squared(snapshot_idx - ref_idx + 1, atom_idx) + displacement_squared
            end do
        end do

        valid_snapshots_count = valid_snapshots_count + 1

        ! Update progress bar
        if (verbose) then
            progress = progress + 1
            if (progress == 1) then
                ! For the first iteration, print progress and move to a new line
                write(*, '(A)') ""  ! Newline after the first print
                write(*, '(A,I3,A)', advance='no') "Progress: [", progress * 100 / M, "%]"                
            else
                ! For subsequent iterations, update the progress on the same line
                write(*, '(A)', advance='no') char(13)  ! Return carriage to overwrite the previous progress
                write(*, '(A,I3,A)', advance='no') "Progress: [", progress * 100 / M, "%]"
            end if
        end if
    end do

    !-------------------------------------------------------------------
    ! Normalize results
    !-------------------------------------------------------------------
    if (valid_snapshots_count > 0) then
        delta_squared = delta_squared / real(valid_snapshots_count, 8)
        ! if (verbose) print *, "Normalization complete"
    else
        print *, "Error: No valid iterations were performed (valid_snapshots_count=0)"
        stop
    end if

    ! if (verbose) print *, "Subroutine compute_delta_squared completed successfully"
end subroutine shifted_msd

subroutine shifted_msd_beads(positions, delta_squared, verbose, nbeads, nsnapshots, natoms, M)
    implicit none

    !-------------------------------------------------------------------
    ! Input arguments
    !-------------------------------------------------------------------
    integer, intent(in) :: nbeads, nsnapshots, natoms, M
    logical, intent(in) :: verbose
    real(8), intent(in) :: positions(3,natoms,nsnapshots,nbeads)

    !-------------------------------------------------------------------
    ! Output argument
    !-------------------------------------------------------------------
    real(8), intent(inout) :: delta_squared(M, natoms)  ! (M, natoms)

    !-------------------------------------------------------------------
    ! Local variables
    !-------------------------------------------------------------------
    integer :: snapshot_idx, atom_idx, ref_idx, b
    real(8), dimension(natoms, 3) :: ref_positions
    real(8) :: displacement_squared
    integer :: valid_snapshots_count

    ! Progress tracking
    integer :: progress = 0
    integer :: last_printed = -1
    real :: start_time, current_time, elapsed, eta
    real :: progress_fraction
    integer :: dt(8)  ! For date_and_time

    !-------------------------------------------------------------------
    ! Input validation
    !-------------------------------------------------------------------
    if (nbeads <= 1) then
        print *, "Error: nbeads must be larger than 1"
        stop
    end if
    if (nsnapshots < 2 * M) then
        print *, "Error: nsnapshots must be at least 2*M"
        stop
    end if
    if (M <= 0) then
        print *, "Error: M must be greater than 0"
        stop
    end if
    if (natoms <= 0) then
        print *, "Error: natoms must be greater than 0"
        stop
    end if

    !-------------------------------------------------------------------
    ! Initialization
    !-------------------------------------------------------------------
    delta_squared(:,:) = 0.0
    valid_snapshots_count = 0
    progress = 0
    call cpu_time(start_time)

    !-------------------------------------------------------------------
    ! Main computation loop
    !-------------------------------------------------------------------
    do b = 1, nbeads
        do ref_idx = 1, M
            ! Reference configuration
            ref_positions = positions(:, :, ref_idx, b)

            do snapshot_idx = ref_idx, ref_idx + M - 1
                if (snapshot_idx > nsnapshots) exit

                do atom_idx = 1, natoms
                    displacement_squared = sum((positions(:, atom_idx, snapshot_idx, b) - &
                                                ref_positions(atom_idx, :))**2)
                    delta_squared(snapshot_idx - ref_idx + 1, atom_idx) = &
                        delta_squared(snapshot_idx - ref_idx + 1, atom_idx) + displacement_squared
                end do
            end do

            valid_snapshots_count = valid_snapshots_count + 1

            ! Update progress bar
            if (verbose) then
                progress = progress + 1
                progress_fraction = real(progress) / real(M * nbeads)

                if (int(progress_fraction*100) /= last_printed) then
                    last_printed = int(progress_fraction*100)

                    ! Elapsed and ETA
                    call cpu_time(current_time)
                    elapsed = current_time - start_time
                    if (progress_fraction > 0.0) then
                        eta = elapsed * (1.0 - progress_fraction) / progress_fraction
                    else
                        eta = 0.0
                    end if

                    ! Current timestamp
                    call date_and_time(values=dt)

                    ! Print progress
                    write(*, '(A)', advance='no') char(13)
                    write(*,'(I4.4,"-",I2.2,"-",I2.2," ",I2.2,":",I2.2,":",I2.2,".",I3.3," Progress: [",I3.3,"%%] ETA: ",F6.1," s")', advance='no') &
                        dt(1), dt(2), dt(3), dt(4), dt(5), dt(6), dt(7), last_printed, eta
                end if
            end if
        end do
    end do

    !-------------------------------------------------------------------
    ! Normalize results
    !-------------------------------------------------------------------
    if (valid_snapshots_count > 0) then
        delta_squared = delta_squared / real(valid_snapshots_count*nbeads, 8)
    else
        print *, "Error: No valid iterations were performed (valid_snapshots_count=0)"
        stop
    end if

    ! Print newline after completion
    if (verbose) write(*,*)
end subroutine shifted_msd_beads
