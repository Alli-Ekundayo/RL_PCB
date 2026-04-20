#!/bin/bash

CPU_ONLY=false
UPDATE_UTILITY_BINARIES=false
SKIP_REPOSITORY_CHECK=false
	date="2026/03/26"

patch_missing_headers() {
    echo "Patching missing <cstdint> headers in $(pwd)..."
    grep -rlE "uint(8|16|32|64)_t" . --include="*.hpp" --include="*.cpp" --include="*.h" --include="*.c" | xargs --no-run-if-empty grep -L "cstdint" | xargs --no-run-if-empty -I {} sed -i '1i #include <cstdint>' {}

    # Patch Makefiles to add boost include/library paths if they exist
    BOOST_INCLUDE=""
    BOOST_LIBDIR=""

    # Check known locations for boost headers
    for dir in /opt/or-tools/include /usr/include /usr/local/include; do
        if [ -f "${dir}/boost/io/ios_state.hpp" ]; then
            BOOST_INCLUDE="${dir}"
            echo "Found boost headers at: ${BOOST_INCLUDE}"
            break
        fi
    done

    # Check known locations for boost_system and boost_filesystem libs
    for dir in /usr/lib/nsight-systems/host-linux-x64 /usr/lib/x86_64-linux-gnu /usr/local/lib /opt/or-tools/lib; do
        if ls "${dir}/libboost_system"* 2>/dev/null | grep -q .; then
            BOOST_LIBDIR="${dir}"
            echo "Found boost libraries at: ${BOOST_LIBDIR}"
            break
        fi
    done

    # Patch Makefiles in current directory tree
    if [ -n "${BOOST_INCLUDE}" ] || [ -n "${BOOST_LIBDIR}" ]; then
        for mk in $(find . -name "Makefile" -maxdepth 3); do
            if grep -q "boost" "${mk}"; then
                if [ -n "${BOOST_INCLUDE}" ] && ! grep -q "or-tools/include" "${mk}"; then
                    sed -i "s|-I ./include|-I ./include -I ${BOOST_INCLUDE}|g" "${mk}"
                    echo "Patched boost include in ${mk}"
                fi
                if [ -n "${BOOST_LIBDIR}" ] && ! grep -q "${BOOST_LIBDIR}" "${mk}"; then
                    sed -i "s|-lboost_system|-L${BOOST_LIBDIR} -lboost_system|g" "${mk}"
                    echo "Patched boost libdir in ${mk}"
                fi
            fi
        done
    fi
}


update_utility_binaries() {
	date="2023/05/06"

	kicadParser_branch=parsing_and_plotting
	SA_PCB_branch=crtYrd_bbox
	pcbRouter_branch=updating_dependencies

	#GIT=https://www.github.com/
	#GIT_USER=lukevassallo
	#GIT=git@gitlab.lukevassallo.com:
	GIT=https://gitlab.lukevassallo.com/
    GIT_USER=luke

	CLEAN_ONLY=false
	CLEAN_BEFORE_BUILD=false
	RUN_PLACER_TESTS=false
	RUN_ROUTER_TESTS=false

	# Set compiler flags for compatibility with older CPUs (i.e., Haswell and newer)
	# Use -march=x86-64-v3 for broader compatibility while still enabling some vectorization
	# For maximum compatibility, use -march=x86-64 (baseline x86-64)
	# Added -w to suppress noisy warnings during automated builds
	export CFLAGS="-std=c++14 -O2 -fPIC -fpermissive -march=x86-64-v3 -w"
	export CXXFLAGS="${CFLAGS}"

	printf "\n"
	printf "  **** Luke Vassallo M.Sc - 02_update_utility_binaries.sh\n"
	printf "   *** Program to to update kicad parsing utility and place and route tols.\n"
	printf "    ** Last modification time %s\n" $date
	printf "\n"
	sleep 1

	print_help() {
		echo "  --clean_only                removes the git repositories and exits."
		echo "  --clean_before_build        removes the git repositories then clones and builds binaries."
		echo "  --run_placer_tests          runs automated tests to verify placer."
		echo "  --run_router_tests          runs automated tests to verify router."
		echo "  --help                      print this help and exit."
	}

    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean_only)
                CLEAN_ONLY=true
                shift   # past argument
                ;;
            --clean_before_build)
                CLEAN_BEFORE_BUILD=true
                shift
                ;;
            --run_placer_tests)
                RUN_PLACER_TESTS=true
                shift
                ;;
            --run_router_tests)
                RUN_ROUTER_TESTS=true
                shift
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            -*|--*)
                echo "Unknown option $1"
                exit 1
                ;;
            *)
                POSITIONAL_ARGS+=($1)
                shift
                ;;
        esac
    done

    if [ -d "bin" ]; then
        cd bin
    else
        mkdir bin && cd bin
    fi

    if [ "$CLEAN_ONLY" = true ] || [ "$CLEAN_BEFORE_BUILD" = true ]; then
        echo -n "Attempting to clean the KicadParser repository ... "
        if [ ! -d "KicadParser" ]; then
            echo "Not found, therefore nothing to clean.";
        else
            echo "Found, deleting."
            rm -fr KicadParser
        fi

        echo -n "Attempting to clean the SA_PCB repository ... "
        if [ ! -d "SA_PCB" ]; then
            echo "Not found, therefore nothing to clean.";
        else
            echo "Found, deleting."
            rm -fr SA_PCB
        fi

        echo -n "Attempting to clean the pcbRouter repository ... "
        if [ ! -d "pcbRouter" ]; then
            echo "Not found, therefore nothing to clean.";
        else
            echo "Found, deleting."
            rm -fr pcbRouter
        fi

        if [ "$CLEAN_ONLY" = true ]; then
            exit 0
        fi
    fi

    echo -n "Building kicad pcb parsing utility. Checking for repository ... "
    ORIGIN=${GIT}${GIT_USER}/kicadParser
    response=$(curl -sL -I -o /dev/null -w "%{http_code}" "$ORIGIN")
    if [[ $response -eq 200 ]] || [ "$SKIP_REPOSITORY_CHECK" = true ]; then        
        echo "Repository exists."
        if [ -d "KicadParser" ]; then
            echo "Found, cleaning"
            cd KicadParser
            make clean
                git pull $ORIGIN ${kicadParser_branch}
            #git submodule update --remote --recursive
        else
            echo "Not found, cloning."
            git clone --branch ${kicadParser_branch} ${ORIGIN} --recurse-submodules KicadParser
            cd KicadParser
        fi
        patch_missing_headers
        # Patch Makefile to use compatible CFLAGS
        sed -i "s/CFLAGS = -std=c++14 -Ofast/CFLAGS = ${CFLAGS}/g" Makefile
        sed -i "s/CFLAGS = -std=c++14 -O3/CFLAGS = ${CFLAGS}/g" pcb/Makefile
        sed -i "s/CFLAGS = -std=c++14 -Ofast/CFLAGS = ${CFLAGS}/g" pcb/netlist_graph/Makefile
        make -j$(nproc)
        cp -v build/kicadParser_test ../kicadParser
        cd ..
    else
        echo "Repository does not exist."
    fi

    echo -n "Building simulated annealing pcb placer. Checking for repository ... "
    ORIGIN=${GIT}${GIT_USER}/SA_PCB
    response=$(curl -sL -I -o /dev/null -w "%{http_code}" "$ORIGIN")
    if [[ $response -eq 200 ]] || [ "$SKIP_REPOSITORY_CHECK" = true ]; then       
        echo "Repository exists."    
        if [ -d "SA_PCB" ]; then
            echo "Found, cleaning"
            cd SA_PCB
            make clean
            git pull ${ORIGIN} ${SA_PCB_branch}
            #git submodule update --remote --recursive
        else
            echo "Not found, cloning."
            git clone --branch ${SA_PCB_branch} ${ORIGIN} --recurse-submodules
            cd SA_PCB
        fi
        patch_missing_headers
        # Patch Makefile to use compatible CFLAGS
        sed -i "s/CFLAGS = -std=c++14 -Ofast/CFLAGS = ${CFLAGS}/g" Makefile
        sed -i "s/CFLAGS = -std=c++14 -O3/CFLAGS = ${CFLAGS}/g" netlist_graph/Makefile
        sed -i "s/CFLAGS = -std=c++14 -O3/CFLAGS = ${CFLAGS}/g" KicadParser/Makefile
        sed -i "s/CFLAGS = -std=c++14 -O3/CFLAGS = ${CFLAGS}/g" KicadParser/pcb/Makefile
        sed -i "s/CFLAGS = -std=c++14 -O3/CFLAGS = ${CFLAGS}/g" KicadParser/pcb/netlist_graph/Makefile
        make -j$(nproc)
        if [ "$RUN_PLACER_TESTS" = true ]; then
            make test_place_excl_power
            make test_place_incl_power
        fi

        #cp -v ./build/sa_placer_test ../bin/sa_placer
        cp -v ./build/sa_placer_test ../sa
        cd ..
    else
        echo "Repository does not exist."
    fi        

    echo -n "Building pcbRouter binary. Checking for repository ... "
    ORIGIN=${GIT}${GIT_USER}/pcbRouter
    response=$(curl -sL -I -o /dev/null -w "%{http_code}" "$ORIGIN")
    if [[ $response -eq 200 ]] || [ "$SKIP_REPOSITORY_CHECK" = true ]; then        
        echo "Repository exists."    
        if [ -d "pcbRouter" ]; then
            echo "Found, cleaning"
            cd pcbRouter
            make clean
                git pull ${ORIGIN} ${pcbRouter_branch}
            #git submodule update --remote --recursive
        else
            echo "Not found, cloning."
            git clone --branch ${pcbRouter_branch} ${ORIGIN} --recurse-submodules
            cd pcbRouter
        fi
        patch_missing_headers
        # Patch Makefile to use compatible CFLAGS
        sed -i "s/CFLAGS = -std=c++14 -Ofast/CFLAGS = ${CFLAGS}/g" Makefile
        sed -i "s/CFLAGS = -std=c++14 -O3/CFLAGS = ${CFLAGS}/g" netlist_graph/Makefile
        sed -i "s/CFLAGS = -std=c++14 -O3/CFLAGS = ${CFLAGS}/g" KicadParser/Makefile
        sed -i "s/CFLAGS = -std=c++14 -O3/CFLAGS = ${CFLAGS}/g" KicadParser/pcb/Makefile
        sed -i "s/CFLAGS = -std=c++14 -O3/CFLAGS = ${CFLAGS}/g" KicadParser/pcb/netlist_graph/Makefile
        make -j$(nproc)
        if [ "$RUN_ROUTER_TESTS" = true ]; then
            make test_route_excl_power
            make test_route_incl_power
        fi

        cp -v build/pcbRouter_test ../pcb_router
        cd ..
    else
        echo "Repository does not exist."
    fi    

    cd ..
}

    printf "\n"
	printf "  **** Luke Vassallo M.Sc - install_tools_and_virtual_environment.sh\n"
	printf "   *** Program to setup the environemnt for RL_PCB and baseline place and route tools.\n"
    printf "\033[32m"       # Green text color
    printf "       RL_PCB is an end-to-end Reinforcement Learning PCB placement methodology.\n"
    printf "\033[0m"        # Black text color
	printf "    ** Last modification time %s\n" $date
	printf "\n"
	sleep 5

print_help() {
    echo "  --cpu_only                  installs the cpu only version of pyTorch."
    echo "  --update_utility_binaries   cleans the git repositories then clones, builds and tests the place and route binaries."
    echo "  --env_only                  (legacy) setup the virtual environment only."
    echo "  --skip-repository-check     skips existance checks when cloning dependent repositories for place and route tools."
    echo "  --help                      print this help and exit."
}
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu_only)
            CPU_ONLY=true
            shift
            ;;
        --update_utility_binaries)
            UPDATE_UTILITY_BINARIES=true
            shift
            ;;
        --env_only)
            # This flag is now the default behavior if --update_utility_binaries is not used.
            # Included for backward compatibility.
            shift
            ;;
        --skip-repository-check)
            SKIP_REPOSITORY_CHECK=true
            shift
            ;;
        -h|--help)
            print_help
            update_utility_binaries --help
            exit 0
            ;;
    esac
done

# Check if python3 exists (Python 3.10+ recommended)
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "Python ${PYTHON_VERSION} is installed."
else
    echo "Python 3 is not installed. Please install python3 (3.10+) and relaunch the script."
    exit 1
fi

source setup.sh

if [ "$UPDATE_UTILITY_BINARIES" == true ]; then
	update_utility_binaries --clean_before_build --run_placer_tests --run_router_tests
    exit 0
fi

if [ ! -d "bin" ]; then
    echo "Installing kicad PCB parsing utility and PCB place and route tools."
    update_utility_binaries --run_placer_tests --run_router_tests
fi

if [ ! -d "venv" ]; then
	echo "Creating virtual environment ..."
	python3 -m venv venv
else
	echo "Virtual environment already exists ..."
fi
source venv/bin/activate

which python
python -c "import sys; print(sys.path)"
python -V

python -m pip install --upgrade pip
python -m pip install --upgrade "setuptools<81"

python -m pip install -r requirements.txt
# Auto-detect CUDA version and install matching PyTorch
if [ "$CPU_ONLY" == true ]; then
	python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
	# Detect CUDA version from nvcc and map to available PyTorch wheel
	if command -v nvcc &>/dev/null; then
		CUDA_VER_RAW=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
		CUDA_MAJOR=$(echo "$CUDA_VER_RAW" | cut -d. -f1)
		CUDA_MINOR=$(echo "$CUDA_VER_RAW" | cut -d. -f2)
		
		# Map CUDA version to available PyTorch wheel (cu118, cu121, cu124, cu126, etc.)
		if [ "$CUDA_MAJOR" -eq 11 ]; then
			PYTORCH_CUDA="cu118"
		elif [ "$CUDA_MAJOR" -eq 12 ]; then
			if [ "$CUDA_MINOR" -le 1 ]; then
				PYTORCH_CUDA="cu121"
			elif [ "$CUDA_MINOR" -le 4 ]; then
				PYTORCH_CUDA="cu124"
			else
				PYTORCH_CUDA="cu126"
			fi
		else
			# For CUDA 13+, try latest available
			PYTORCH_CUDA="cu126"
		fi
		
		echo "Detected CUDA version: ${CUDA_VER_RAW} → using ${PYTORCH_CUDA} PyTorch wheel"
		python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"
	else
		echo "nvcc not found, falling back to CUDA 12.4 PyTorch wheel"
		python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
	fi
fi

#python -m pip install matplotlib numpy==1.23.3 opencv-python gym pyglet optuna tensorboard reportlab==3.6.13 py-cpuinfo psutil pandas seaborn pynvml plotly moviepy

#python -m pip install -U kaleido

python -m pip install ${RL_PCB}/lib/pcb_netlist_graph-0.0.1-py3-none-any.whl
python -m pip install ${RL_PCB}/lib/pcb_file_io-0.0.1-py3-none-any.whl

python ${RL_PCB}/tests/00_verify_machine_setup/test_setup.py
