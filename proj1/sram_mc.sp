*test the normal SRAM_cell during reading status
*45nm technology
*test using hspice C-2009.09-SP1


* BSIM4 model file
.include "./45nm_bsim4.txt"
.param nvth1 = vthn1
.param nvth2 = vthn2
.param nvth3 = vthn3
.param nvth4 = vthn4

.param pvth1 = vthp1
.param pvth2 = vthp2

.param lmda = 0.05u

.GLOBAL VDD!
.PARAM VDD = 1.3


.PARAM BITCAP = 1e-12
CBL BLB 0 BITCAP
CBLB BL 0 BITCAP

.option post=1

* one inverter
MPL data ndata vdd_node vdd_node  pmos1 l='2*lmda' w='6.5*lmda' 
MNL data ndata 0 0 nmos1  l='2*lmda' w='7.5*lmda' 

* one inverter
MPR ndata data vdd_node vdd_node pmos2  l='2*lmda' w='6.5*lmda' 
MNR ndata data 0 0 nmos2  l='2*lmda' w='7.5*lmda' 

* access transistors
MNAL bit word data 0 nmos3 l='2*lmda' w='7.5*lmda' 
MNAR nbit word ndata 0 nmos4 l='2*lmda' w='7.5*lmda' 

* Power supply
vdd vdd_node 0 DC=VDD

* SRAM control signals
v_wd word 0 DC=VDD

* Initial status
.IC V(bit) = VDD
.IC V(nbit) = VDD

* SRAM Initial Value
.nodeset v(data)=VDD
.nodeset v(ndata)=0



* Simulation/Anaylsis settings
.temp 27
.op 
.tran 0.1ps 25ps SWEEP DATA=data


*** .DATA goes here
.include "./sweep_data_mc"

* Ouptuts
.print v(bit) v(nbit) 
.option post=1


.end