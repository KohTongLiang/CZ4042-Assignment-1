??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
DenseBincount
input"Tidx
size"Tidx
weights"T
output"T"
Tidxtype:
2	"
Ttype:
2	"
binary_outputbool( 
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name141*
value_dtype0	
}
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name
table_64*
value_dtype0	
m
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name251*
value_dtype0	
?
MutableHashTable_1MutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_174*
value_dtype0	
m
hash_table_2HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name361*
value_dtype0	
?
MutableHashTable_2MutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_284*
value_dtype0	
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
d
mean_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean_1
]
mean_1/Read/ReadVariableOpReadVariableOpmean_1*
_output_shapes
:*
dtype0
l

variance_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance_1
e
variance_1/Read/ReadVariableOpReadVariableOp
variance_1*
_output_shapes
:*
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0	
d
mean_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean_2
]
mean_2/Read/ReadVariableOpReadVariableOpmean_2*
_output_shapes
:*
dtype0
l

variance_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance_2
e
variance_2/Read/ReadVariableOpReadVariableOp
variance_2*
_output_shapes
:*
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0	
d
mean_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean_3
]
mean_3/Read/ReadVariableOpReadVariableOpmean_3*
_output_shapes
:*
dtype0
l

variance_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance_3
e
variance_3/Read/ReadVariableOpReadVariableOp
variance_3*
_output_shapes
:*
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0	
d
mean_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean_4
]
mean_4/Read/ReadVariableOpReadVariableOpmean_4*
_output_shapes
:*
dtype0
l

variance_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance_4
e
variance_4/Read/ReadVariableOpReadVariableOp
variance_4*
_output_shapes
:*
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0	
d
mean_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean_5
]
mean_5/Read/ReadVariableOpReadVariableOpmean_5*
_output_shapes
:*
dtype0
l

variance_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance_5
e
variance_5/Read/ReadVariableOpReadVariableOp
variance_5*
_output_shapes
:*
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0	
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Q
*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:Q
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:
*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_9
[
count_9/Read/ReadVariableOpReadVariableOpcount_9*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Q
*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:Q
*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:
*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Q
*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:Q
*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:
*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
\
Const_3Const*
_output_shapes

:*
dtype0*
valueB*?r?B
\
Const_4Const*
_output_shapes

:*
dtype0*
valueB*=rD
\
Const_5Const*
_output_shapes

:*
dtype0*
valueB*}P?
\
Const_6Const*
_output_shapes

:*
dtype0*
valueB*d^>
\
Const_7Const*
_output_shapes

:*
dtype0*
valueB*$?3A
\
Const_8Const*
_output_shapes

:*
dtype0*
valueB*?z?A
\
Const_9Const*
_output_shapes

:*
dtype0*
valueB*_?<
]
Const_10Const*
_output_shapes

:*
dtype0*
valueB*j??7
]
Const_11Const*
_output_shapes

:*
dtype0*
valueB*<r?;
]
Const_12Const*
_output_shapes

:*
dtype0*
valueB*???9
]
Const_13Const*
_output_shapes

:*
dtype0*
valueB*z?B
]
Const_14Const*
_output_shapes

:*
dtype0*
valueB*=?'C
J
Const_15Const*
_output_shapes
: *
dtype0	*
value	B	 R 
J
Const_16Const*
_output_shapes
: *
dtype0	*
value	B	 R 
J
Const_17Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Const_18Const*
_output_shapes
:*
dtype0	*u
valuelBj	"`       
                            	                                                 
?
Const_19Const*
_output_shapes
:*
dtype0	*u
valuelBj	"`                                                        	       
                     
?
Const_20Const*
_output_shapes
:+*
dtype0*?
value?B?+B4 ROOM, Model AB5 ROOM, ImprovedB3 ROOM, New GenerationB3 ROOM, ImprovedB3 ROOM, Model AB4 ROOM, Premium ApartmentB4 ROOM, New GenerationBEXECUTIVE, ApartmentB5 ROOM, Premium ApartmentB4 ROOM, SimplifiedBEXECUTIVE, MaisonetteB3 ROOM, StandardB5 ROOM, Model AB4 ROOM, Model A2B4 ROOM, ImprovedB3 ROOM, SimplifiedB5 ROOM, StandardBEXECUTIVE, Premium ApartmentB2 ROOM, Model AB5 ROOM, DBSSB4 ROOM, DBSSB3 ROOM, Premium ApartmentB2 ROOM, StandardB2 ROOM, ImprovedB3 ROOM, DBSSB5 ROOM, Model A-MaisonetteB4 ROOM, Type S1B5 ROOM, Adjoined flatB5 ROOM, Type S2BEXECUTIVE, Adjoined flatB3 ROOM, TerraceB"MULTI-GENERATION, Multi GenerationB1 ROOM, ImprovedB4 ROOM, StandardB4 ROOM, Premium Apartment LoftB2 ROOM, Premium ApartmentB5 ROOM, Improved-MaisonetteB4 ROOM, Adjoined flatBEXECUTIVE, Premium MaisonetteB5 ROOM, Premium Apartment LoftB2 ROOM, 2-roomB4 ROOM, TerraceB2 ROOM, DBSS
?
Const_21Const*
_output_shapes
:+*
dtype0	*?
value?B?	+"?                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       
?
Const_22Const*
_output_shapes
:*
dtype0*?
value?B?B04 TO 06B07 TO 09B10 TO 12B01 TO 03B13 TO 15B16 TO 18B19 TO 21B22 TO 24B25 TO 27B28 TO 30B31 TO 33B37 TO 39B34 TO 36B40 TO 42B46 TO 48B43 TO 45B49 TO 51
?
Const_23Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                        	       
                                                        
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_18Const_19*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_295255
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_295260
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_1Const_20Const_21*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_295268
?
PartitionedCall_1PartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_295273
?
StatefulPartitionedCall_2StatefulPartitionedCallhash_table_2Const_22Const_23*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_295281
?
PartitionedCall_2PartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_295286
?
NoOpNoOp^PartitionedCall^PartitionedCall_1^PartitionedCall_2^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0	*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?
AMutableHashTable_1_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable_1*
Tkeys0*
Tvalues0	*%
_class
loc:@MutableHashTable_1*
_output_shapes

::
?
AMutableHashTable_2_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable_2*
Tkeys0*
Tvalues0	*%
_class
loc:@MutableHashTable_2*
_output_shapes

::
?A
Const_24Const"/device:CPU:0*
_output_shapes
: *
dtype0*?A
value?AB?A B?A
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-0

layer-9
layer_with_weights-1
layer-10
layer_with_weights-2
layer-11
layer_with_weights-3
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
 
 
 
 
 
 
 
 
3
lookup_table
token_counts
	keras_api
3
lookup_table
 token_counts
!	keras_api
3
"lookup_table
#token_counts
$	keras_api
?
%
_keep_axis
&_reduce_axis
'_reduce_axis_mask
(_broadcast_shape
)mean
)
adapt_mean
*variance
*adapt_variance
	+count
,	keras_api
?
-
_keep_axis
._reduce_axis
/_reduce_axis_mask
0_broadcast_shape
1mean
1
adapt_mean
2variance
2adapt_variance
	3count
4	keras_api
?
5
_keep_axis
6_reduce_axis
7_reduce_axis_mask
8_broadcast_shape
9mean
9
adapt_mean
:variance
:adapt_variance
	;count
<	keras_api
?
=
_keep_axis
>_reduce_axis
?_reduce_axis_mask
@_broadcast_shape
Amean
A
adapt_mean
Bvariance
Badapt_variance
	Ccount
D	keras_api
?
E
_keep_axis
F_reduce_axis
G_reduce_axis_mask
H_broadcast_shape
Imean
I
adapt_mean
Jvariance
Jadapt_variance
	Kcount
L	keras_api
?
M
_keep_axis
N_reduce_axis
O_reduce_axis_mask
P_broadcast_shape
Qmean
Q
adapt_mean
Rvariance
Radapt_variance
	Scount
T	keras_api
R
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
h

Ykernel
Zbias
[regularization_losses
\	variables
]trainable_variables
^	keras_api
h

_kernel
`bias
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
?
eiter

fbeta_1

gbeta_2
	hdecay
ilearning_rateYm?Zm?_m?`m?Yv?Zv?_v?`v?
 
?
)3
*4
+5
16
27
38
99
:10
;11
A12
B13
C14
I15
J16
K17
Q18
R19
S20
Y21
Z22
_23
`24

Y0
Z1
_2
`3
?
jlayer_metrics

klayers
regularization_losses
lnon_trainable_variables
mmetrics
	variables
nlayer_regularization_losses
trainable_variables
 

o_initializer
><
table3layer_with_weights-0/token_counts/.ATTRIBUTES/table
 

p_initializer
><
table3layer_with_weights-1/token_counts/.ATTRIBUTES/table
 

q_initializer
><
table3layer_with_weights-2/token_counts/.ATTRIBUTES/table
 
 
 
 
 
NL
VARIABLE_VALUEmean4layer_with_weights-3/mean/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvariance8layer_with_weights-3/variance/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcount5layer_with_weights-3/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_14layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_18layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_15layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_24layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_28layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_25layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_34layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_38layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_35layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_44layer_with_weights-7/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_48layer_with_weights-7/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_45layer_with_weights-7/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_54layer_with_weights-8/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_58layer_with_weights-8/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_55layer_with_weights-8/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
?
rlayer_metrics

slayers
Uregularization_losses
tnon_trainable_variables
umetrics
V	variables
vlayer_regularization_losses
Wtrainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Y0
Z1

Y0
Z1
?
wlayer_metrics

xlayers
[regularization_losses
ynon_trainable_variables
zmetrics
\	variables
{layer_regularization_losses
]trainable_variables
[Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

_0
`1

_0
`1
?
|layer_metrics

}layers
aregularization_losses
~non_trainable_variables
metrics
b	variables
 ?layer_regularization_losses
ctrainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
?
)3
*4
+5
16
27
38
99
:10
;11
A12
B13
C14
I15
J16
K17
Q18
R19
S20
 
?0
?1
?2
?3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_74keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_84keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_94keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_1/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_1/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_1/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_1/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
!serving_default_degree_centralityPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_dist_to_dhobyPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
#serving_default_dist_to_nearest_stnPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
&serving_default_eigenvector_centralityPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_flat_model_typePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_floor_area_sqmPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
x
serving_default_monthPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
%serving_default_remaining_lease_yearsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_storey_rangePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_3StatefulPartitionedCall!serving_default_degree_centralityserving_default_dist_to_dhoby#serving_default_dist_to_nearest_stn&serving_default_eigenvector_centralityserving_default_flat_model_typeserving_default_floor_area_sqmserving_default_month%serving_default_remaining_lease_yearsserving_default_storey_range
hash_tableConsthash_table_1Const_1hash_table_2Const_2Const_3Const_4Const_5Const_6Const_7Const_8Const_9Const_10Const_11Const_12Const_13Const_14dense/kernel
dense/biasdense_1/kerneldense_1/bias**
Tin#
!2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_294617
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filename?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1AMutableHashTable_1_lookup_table_export_values/LookupTableExportV2CMutableHashTable_1_lookup_table_export_values/LookupTableExportV2:1AMutableHashTable_2_lookup_table_export_values/LookupTableExportV2CMutableHashTable_2_lookup_table_export_values/LookupTableExportV2:1mean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOpmean_1/Read/ReadVariableOpvariance_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpmean_2/Read/ReadVariableOpvariance_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpmean_3/Read/ReadVariableOpvariance_3/Read/ReadVariableOpcount_3/Read/ReadVariableOpmean_4/Read/ReadVariableOpvariance_4/Read/ReadVariableOpcount_4/Read/ReadVariableOpmean_5/Read/ReadVariableOpvariance_5/Read/ReadVariableOpcount_5/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount_6/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_8/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_9/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst_24*>
Tin7
523											*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_295494
?
StatefulPartitionedCall_5StatefulPartitionedCallsaver_filenameMutableHashTableMutableHashTable_1MutableHashTable_2meanvariancecountmean_1
variance_1count_1mean_2
variance_2count_2mean_3
variance_3count_3mean_4
variance_4count_4mean_5
variance_5count_5dense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount_6total_1count_7total_2count_8total_3count_9Adam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_295642??
?
?
__inference_adapt_step_7691
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNextq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shape?
ReshapeReshapeIteratorGetNext:components:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshape?
UniqueWithCountsUniqueWithCountsReshape:output:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	2
UniqueWithCounts?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:2*
(None_lookup_table_find/LookupTableFindV2?
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2
add?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2.
,None_lookup_table_insert/LookupTableInsertV2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
??
?

A__inference_model_layer_call_and_return_conditional_losses_294186

inputs	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8C
?integer_lookup_hash_table_lookup_lookuptablefindv2_table_handleD
@integer_lookup_hash_table_lookup_lookuptablefindv2_default_value	B
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	D
@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleE
Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	
normalization_sub_y
normalization_sqrt_x
normalization_1_sub_y
normalization_1_sqrt_x
normalization_2_sub_y
normalization_2_sqrt_x
normalization_3_sub_y
normalization_3_sqrt_x
normalization_4_sub_y
normalization_4_sqrt_x
normalization_5_sub_y
normalization_5_sqrt_x
dense_294175:Q

dense_294177:
 
dense_1_294180:

dense_1_294182:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?2integer_lookup/hash_table_Lookup/LookupTableFindV2?1string_lookup/hash_table_Lookup/LookupTableFindV2?3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
2integer_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2?integer_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs@integer_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????24
2integer_lookup/hash_table_Lookup/LookupTableFindV2?
integer_lookup/IdentityIdentity;integer_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup/Identity?
integer_lookup/bincount/ShapeShape integer_lookup/Identity:output:0*
T0	*
_output_shapes
:2
integer_lookup/bincount/Shape?
integer_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
integer_lookup/bincount/Const?
integer_lookup/bincount/ProdProd&integer_lookup/bincount/Shape:output:0&integer_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2
integer_lookup/bincount/Prod?
!integer_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!integer_lookup/bincount/Greater/y?
integer_lookup/bincount/GreaterGreater%integer_lookup/bincount/Prod:output:0*integer_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2!
integer_lookup/bincount/Greater?
integer_lookup/bincount/CastCast#integer_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
integer_lookup/bincount/Cast?
integer_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
integer_lookup/bincount/Const_1?
integer_lookup/bincount/MaxMax integer_lookup/Identity:output:0(integer_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/Max?
integer_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
integer_lookup/bincount/add/y?
integer_lookup/bincount/addAddV2$integer_lookup/bincount/Max:output:0&integer_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/add?
integer_lookup/bincount/mulMul integer_lookup/bincount/Cast:y:0integer_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/mul?
!integer_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!integer_lookup/bincount/minlength?
integer_lookup/bincount/MaximumMaximum*integer_lookup/bincount/minlength:output:0integer_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2!
integer_lookup/bincount/Maximum?
!integer_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!integer_lookup/bincount/maxlength?
integer_lookup/bincount/MinimumMinimum*integer_lookup/bincount/maxlength:output:0#integer_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2!
integer_lookup/bincount/Minimum?
integer_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2!
integer_lookup/bincount/Const_2?
%integer_lookup/bincount/DenseBincountDenseBincount integer_lookup/Identity:output:0#integer_lookup/bincount/Minimum:z:0(integer_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2'
%integer_lookup/bincount/DenseBincount?
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs_1?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????23
1string_lookup/hash_table_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup/Identity?
string_lookup/bincount/ShapeShapestring_lookup/Identity:output:0*
T0	*
_output_shapes
:2
string_lookup/bincount/Shape?
string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
string_lookup/bincount/Const?
string_lookup/bincount/ProdProd%string_lookup/bincount/Shape:output:0%string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2
string_lookup/bincount/Prod?
 string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 string_lookup/bincount/Greater/y?
string_lookup/bincount/GreaterGreater$string_lookup/bincount/Prod:output:0)string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2 
string_lookup/bincount/Greater?
string_lookup/bincount/CastCast"string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
string_lookup/bincount/Cast?
string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
string_lookup/bincount/Const_1?
string_lookup/bincount/MaxMaxstring_lookup/Identity:output:0'string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/Max~
string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
string_lookup/bincount/add/y?
string_lookup/bincount/addAddV2#string_lookup/bincount/Max:output:0%string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/add?
string_lookup/bincount/mulMulstring_lookup/bincount/Cast:y:0string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/mul?
 string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2"
 string_lookup/bincount/minlength?
string_lookup/bincount/MaximumMaximum)string_lookup/bincount/minlength:output:0string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2 
string_lookup/bincount/Maximum?
 string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2"
 string_lookup/bincount/maxlength?
string_lookup/bincount/MinimumMinimum)string_lookup/bincount/maxlength:output:0"string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2 
string_lookup/bincount/Minimum?
string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2 
string_lookup/bincount/Const_2?
$string_lookup/bincount/DenseBincountDenseBincountstring_lookup/Identity:output:0"string_lookup/bincount/Minimum:z:0'string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2&
$string_lookup/bincount/DenseBincount?
3string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleinputs_2Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity<string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_1/Identity?
string_lookup_1/bincount/ShapeShape!string_lookup_1/Identity:output:0*
T0	*
_output_shapes
:2 
string_lookup_1/bincount/Shape?
string_lookup_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
string_lookup_1/bincount/Const?
string_lookup_1/bincount/ProdProd'string_lookup_1/bincount/Shape:output:0'string_lookup_1/bincount/Const:output:0*
T0*
_output_shapes
: 2
string_lookup_1/bincount/Prod?
"string_lookup_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"string_lookup_1/bincount/Greater/y?
 string_lookup_1/bincount/GreaterGreater&string_lookup_1/bincount/Prod:output:0+string_lookup_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2"
 string_lookup_1/bincount/Greater?
string_lookup_1/bincount/CastCast$string_lookup_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
string_lookup_1/bincount/Cast?
 string_lookup_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 string_lookup_1/bincount/Const_1?
string_lookup_1/bincount/MaxMax!string_lookup_1/Identity:output:0)string_lookup_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/Max?
string_lookup_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2 
string_lookup_1/bincount/add/y?
string_lookup_1/bincount/addAddV2%string_lookup_1/bincount/Max:output:0'string_lookup_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/add?
string_lookup_1/bincount/mulMul!string_lookup_1/bincount/Cast:y:0 string_lookup_1/bincount/add:z:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/mul?
"string_lookup_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"string_lookup_1/bincount/minlength?
 string_lookup_1/bincount/MaximumMaximum+string_lookup_1/bincount/minlength:output:0 string_lookup_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2"
 string_lookup_1/bincount/Maximum?
"string_lookup_1/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"string_lookup_1/bincount/maxlength?
 string_lookup_1/bincount/MinimumMinimum+string_lookup_1/bincount/maxlength:output:0$string_lookup_1/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2"
 string_lookup_1/bincount/Minimum?
 string_lookup_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2"
 string_lookup_1/bincount/Const_2?
&string_lookup_1/bincount/DenseBincountDenseBincount!string_lookup_1/Identity:output:0$string_lookup_1/bincount/Minimum:z:0)string_lookup_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2(
&string_lookup_1/bincount/DenseBincount~
normalization/subSubinputs_3normalization_sub_y*
T0*'
_output_shapes
:?????????2
normalization/subo
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
normalization_1/subSubinputs_4normalization_1_sub_y*
T0*'
_output_shapes
:?????????2
normalization_1/subu
normalization_1/SqrtSqrtnormalization_1_sqrt_x*
T0*
_output_shapes

:2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_1/Maximum/y?
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_1/Maximum?
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_1/truediv?
normalization_2/subSubinputs_5normalization_2_sub_y*
T0*'
_output_shapes
:?????????2
normalization_2/subu
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes

:2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_2/Maximum/y?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_2/Maximum?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_2/truediv?
normalization_3/subSubinputs_6normalization_3_sub_y*
T0*'
_output_shapes
:?????????2
normalization_3/subu
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_3/truediv?
normalization_4/subSubinputs_7normalization_4_sub_y*
T0*'
_output_shapes
:?????????2
normalization_4/subu
normalization_4/SqrtSqrtnormalization_4_sqrt_x*
T0*
_output_shapes

:2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_4/Maximum/y?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_4/Maximum?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_4/truediv?
normalization_5/subSubinputs_8normalization_5_sub_y*
T0*'
_output_shapes
:?????????2
normalization_5/subu
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes

:2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_5/Maximum/y?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_5/Maximum?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_5/truediv?
concatenate/PartitionedCallPartitionedCall.integer_lookup/bincount/DenseBincount:output:0-string_lookup/bincount/DenseBincount:output:0/string_lookup_1/bincount/DenseBincount:output:0normalization/truediv:z:0normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Q* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2938712
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_294175dense_294177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2938842
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_294180dense_1_294182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2939002!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall3^integer_lookup/hash_table_Lookup/LookupTableFindV22^string_lookup/hash_table_Lookup/LookupTableFindV24^string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2h
2integer_lookup/hash_table_Lookup/LookupTableFindV22integer_lookup/hash_table_Lookup/LookupTableFindV22f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV22j
3string_lookup_1/hash_table_Lookup/LookupTableFindV23string_lookup_1/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
&__inference_model_layer_call_fn_293954	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_years
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17:Q


unknown_18:


unknown_19:


unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmonthflat_model_typestorey_rangefloor_area_sqmdist_to_nearest_stndist_to_dhobydegree_centralityeigenvector_centralityremaining_lease_yearsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20**
Tin#
!2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2939072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namemonth:XT
'
_output_shapes
:?????????
)
_user_specified_nameflat_model_type:UQ
'
_output_shapes
:?????????
&
_user_specified_namestorey_range:WS
'
_output_shapes
:?????????
(
_user_specified_namefloor_area_sqm:\X
'
_output_shapes
:?????????
-
_user_specified_namedist_to_nearest_stn:VR
'
_output_shapes
:?????????
'
_user_specified_namedist_to_dhoby:ZV
'
_output_shapes
:?????????
+
_user_specified_namedegree_centrality:_[
'
_output_shapes
:?????????
0
_user_specified_nameeigenvector_centrality:^Z
'
_output_shapes
:?????????
/
_user_specified_nameremaining_lease_years:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
&__inference_model_layer_call_fn_294290	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_years
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17:Q


unknown_18:


unknown_19:


unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmonthflat_model_typestorey_rangefloor_area_sqmdist_to_nearest_stndist_to_dhobydegree_centralityeigenvector_centralityremaining_lease_yearsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20**
Tin#
!2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2941862
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namemonth:XT
'
_output_shapes
:?????????
)
_user_specified_nameflat_model_type:UQ
'
_output_shapes
:?????????
&
_user_specified_namestorey_range:WS
'
_output_shapes
:?????????
(
_user_specified_namefloor_area_sqm:\X
'
_output_shapes
:?????????
-
_user_specified_namedist_to_nearest_stn:VR
'
_output_shapes
:?????????
'
_user_specified_namedist_to_dhoby:ZV
'
_output_shapes
:?????????
+
_user_specified_namedegree_centrality:_[
'
_output_shapes
:?????????
0
_user_specified_nameeigenvector_centrality:^Z
'
_output_shapes
:?????????
/
_user_specified_nameremaining_lease_years:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
__inference_adapt_step_7677
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNextq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shape?
ReshapeReshapeIteratorGetNext:components:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshape?
UniqueWithCountsUniqueWithCountsReshape:output:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	2
UniqueWithCounts?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:2*
(None_lookup_table_find/LookupTableFindV2?
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2
add?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2.
,None_lookup_table_insert/LookupTableInsertV2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?

?
A__inference_dense_layer_call_and_return_conditional_losses_295039

inputs0
matmul_readvariableop_resource:Q
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Q
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Q: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Q
 
_user_specified_nameinputs
?
?
(__inference_dense_1_layer_call_fn_295067

inputs
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2939002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
ɠ
?
A__inference_model_layer_call_and_return_conditional_losses_294752
inputs_0	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8C
?integer_lookup_hash_table_lookup_lookuptablefindv2_table_handleD
@integer_lookup_hash_table_lookup_lookuptablefindv2_default_value	B
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	D
@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleE
Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	
normalization_sub_y
normalization_sqrt_x
normalization_1_sub_y
normalization_1_sqrt_x
normalization_2_sub_y
normalization_2_sqrt_x
normalization_3_sub_y
normalization_3_sqrt_x
normalization_4_sub_y
normalization_4_sqrt_x
normalization_5_sub_y
normalization_5_sqrt_x6
$dense_matmul_readvariableop_resource:Q
3
%dense_biasadd_readvariableop_resource:
8
&dense_1_matmul_readvariableop_resource:
5
'dense_1_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?2integer_lookup/hash_table_Lookup/LookupTableFindV2?1string_lookup/hash_table_Lookup/LookupTableFindV2?3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
2integer_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2?integer_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs_0@integer_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????24
2integer_lookup/hash_table_Lookup/LookupTableFindV2?
integer_lookup/IdentityIdentity;integer_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup/Identity?
integer_lookup/bincount/ShapeShape integer_lookup/Identity:output:0*
T0	*
_output_shapes
:2
integer_lookup/bincount/Shape?
integer_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
integer_lookup/bincount/Const?
integer_lookup/bincount/ProdProd&integer_lookup/bincount/Shape:output:0&integer_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2
integer_lookup/bincount/Prod?
!integer_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!integer_lookup/bincount/Greater/y?
integer_lookup/bincount/GreaterGreater%integer_lookup/bincount/Prod:output:0*integer_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2!
integer_lookup/bincount/Greater?
integer_lookup/bincount/CastCast#integer_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
integer_lookup/bincount/Cast?
integer_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
integer_lookup/bincount/Const_1?
integer_lookup/bincount/MaxMax integer_lookup/Identity:output:0(integer_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/Max?
integer_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
integer_lookup/bincount/add/y?
integer_lookup/bincount/addAddV2$integer_lookup/bincount/Max:output:0&integer_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/add?
integer_lookup/bincount/mulMul integer_lookup/bincount/Cast:y:0integer_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/mul?
!integer_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!integer_lookup/bincount/minlength?
integer_lookup/bincount/MaximumMaximum*integer_lookup/bincount/minlength:output:0integer_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2!
integer_lookup/bincount/Maximum?
!integer_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!integer_lookup/bincount/maxlength?
integer_lookup/bincount/MinimumMinimum*integer_lookup/bincount/maxlength:output:0#integer_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2!
integer_lookup/bincount/Minimum?
integer_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2!
integer_lookup/bincount/Const_2?
%integer_lookup/bincount/DenseBincountDenseBincount integer_lookup/Identity:output:0#integer_lookup/bincount/Minimum:z:0(integer_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2'
%integer_lookup/bincount/DenseBincount?
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs_1?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????23
1string_lookup/hash_table_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup/Identity?
string_lookup/bincount/ShapeShapestring_lookup/Identity:output:0*
T0	*
_output_shapes
:2
string_lookup/bincount/Shape?
string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
string_lookup/bincount/Const?
string_lookup/bincount/ProdProd%string_lookup/bincount/Shape:output:0%string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2
string_lookup/bincount/Prod?
 string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 string_lookup/bincount/Greater/y?
string_lookup/bincount/GreaterGreater$string_lookup/bincount/Prod:output:0)string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2 
string_lookup/bincount/Greater?
string_lookup/bincount/CastCast"string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
string_lookup/bincount/Cast?
string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
string_lookup/bincount/Const_1?
string_lookup/bincount/MaxMaxstring_lookup/Identity:output:0'string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/Max~
string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
string_lookup/bincount/add/y?
string_lookup/bincount/addAddV2#string_lookup/bincount/Max:output:0%string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/add?
string_lookup/bincount/mulMulstring_lookup/bincount/Cast:y:0string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/mul?
 string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2"
 string_lookup/bincount/minlength?
string_lookup/bincount/MaximumMaximum)string_lookup/bincount/minlength:output:0string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2 
string_lookup/bincount/Maximum?
 string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2"
 string_lookup/bincount/maxlength?
string_lookup/bincount/MinimumMinimum)string_lookup/bincount/maxlength:output:0"string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2 
string_lookup/bincount/Minimum?
string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2 
string_lookup/bincount/Const_2?
$string_lookup/bincount/DenseBincountDenseBincountstring_lookup/Identity:output:0"string_lookup/bincount/Minimum:z:0'string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2&
$string_lookup/bincount/DenseBincount?
3string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleinputs_2Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity<string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_1/Identity?
string_lookup_1/bincount/ShapeShape!string_lookup_1/Identity:output:0*
T0	*
_output_shapes
:2 
string_lookup_1/bincount/Shape?
string_lookup_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
string_lookup_1/bincount/Const?
string_lookup_1/bincount/ProdProd'string_lookup_1/bincount/Shape:output:0'string_lookup_1/bincount/Const:output:0*
T0*
_output_shapes
: 2
string_lookup_1/bincount/Prod?
"string_lookup_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"string_lookup_1/bincount/Greater/y?
 string_lookup_1/bincount/GreaterGreater&string_lookup_1/bincount/Prod:output:0+string_lookup_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2"
 string_lookup_1/bincount/Greater?
string_lookup_1/bincount/CastCast$string_lookup_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
string_lookup_1/bincount/Cast?
 string_lookup_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 string_lookup_1/bincount/Const_1?
string_lookup_1/bincount/MaxMax!string_lookup_1/Identity:output:0)string_lookup_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/Max?
string_lookup_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2 
string_lookup_1/bincount/add/y?
string_lookup_1/bincount/addAddV2%string_lookup_1/bincount/Max:output:0'string_lookup_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/add?
string_lookup_1/bincount/mulMul!string_lookup_1/bincount/Cast:y:0 string_lookup_1/bincount/add:z:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/mul?
"string_lookup_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"string_lookup_1/bincount/minlength?
 string_lookup_1/bincount/MaximumMaximum+string_lookup_1/bincount/minlength:output:0 string_lookup_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2"
 string_lookup_1/bincount/Maximum?
"string_lookup_1/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"string_lookup_1/bincount/maxlength?
 string_lookup_1/bincount/MinimumMinimum+string_lookup_1/bincount/maxlength:output:0$string_lookup_1/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2"
 string_lookup_1/bincount/Minimum?
 string_lookup_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2"
 string_lookup_1/bincount/Const_2?
&string_lookup_1/bincount/DenseBincountDenseBincount!string_lookup_1/Identity:output:0$string_lookup_1/bincount/Minimum:z:0)string_lookup_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2(
&string_lookup_1/bincount/DenseBincount~
normalization/subSubinputs_3normalization_sub_y*
T0*'
_output_shapes
:?????????2
normalization/subo
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
normalization_1/subSubinputs_4normalization_1_sub_y*
T0*'
_output_shapes
:?????????2
normalization_1/subu
normalization_1/SqrtSqrtnormalization_1_sqrt_x*
T0*
_output_shapes

:2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_1/Maximum/y?
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_1/Maximum?
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_1/truediv?
normalization_2/subSubinputs_5normalization_2_sub_y*
T0*'
_output_shapes
:?????????2
normalization_2/subu
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes

:2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_2/Maximum/y?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_2/Maximum?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_2/truediv?
normalization_3/subSubinputs_6normalization_3_sub_y*
T0*'
_output_shapes
:?????????2
normalization_3/subu
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_3/truediv?
normalization_4/subSubinputs_7normalization_4_sub_y*
T0*'
_output_shapes
:?????????2
normalization_4/subu
normalization_4/SqrtSqrtnormalization_4_sqrt_x*
T0*
_output_shapes

:2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_4/Maximum/y?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_4/Maximum?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_4/truediv?
normalization_5/subSubinputs_8normalization_5_sub_y*
T0*'
_output_shapes
:?????????2
normalization_5/subu
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes

:2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_5/Maximum/y?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_5/Maximum?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_5/truedivt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2.integer_lookup/bincount/DenseBincount:output:0-string_lookup/bincount/DenseBincount:output:0/string_lookup_1/bincount/DenseBincount:output:0normalization/truediv:z:0normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0 concatenate/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????Q2
concatenate/concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:Q
*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdds
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp3^integer_lookup/hash_table_Lookup/LookupTableFindV22^string_lookup/hash_table_Lookup/LookupTableFindV24^string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2h
2integer_lookup/hash_table_Lookup/LookupTableFindV22integer_lookup/hash_table_Lookup/LookupTableFindV22f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV22j
3string_lookup_1/hash_table_Lookup/LookupTableFindV23string_lookup_1/hash_table_Lookup/LookupTableFindV2:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
??
?

A__inference_model_layer_call_and_return_conditional_losses_294552	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_yearsC
?integer_lookup_hash_table_lookup_lookuptablefindv2_table_handleD
@integer_lookup_hash_table_lookup_lookuptablefindv2_default_value	B
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	D
@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleE
Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	
normalization_sub_y
normalization_sqrt_x
normalization_1_sub_y
normalization_1_sqrt_x
normalization_2_sub_y
normalization_2_sqrt_x
normalization_3_sub_y
normalization_3_sqrt_x
normalization_4_sub_y
normalization_4_sqrt_x
normalization_5_sub_y
normalization_5_sqrt_x
dense_294541:Q

dense_294543:
 
dense_1_294546:

dense_1_294548:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?2integer_lookup/hash_table_Lookup/LookupTableFindV2?1string_lookup/hash_table_Lookup/LookupTableFindV2?3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
2integer_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2?integer_lookup_hash_table_lookup_lookuptablefindv2_table_handlemonth@integer_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????24
2integer_lookup/hash_table_Lookup/LookupTableFindV2?
integer_lookup/IdentityIdentity;integer_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup/Identity?
integer_lookup/bincount/ShapeShape integer_lookup/Identity:output:0*
T0	*
_output_shapes
:2
integer_lookup/bincount/Shape?
integer_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
integer_lookup/bincount/Const?
integer_lookup/bincount/ProdProd&integer_lookup/bincount/Shape:output:0&integer_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2
integer_lookup/bincount/Prod?
!integer_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!integer_lookup/bincount/Greater/y?
integer_lookup/bincount/GreaterGreater%integer_lookup/bincount/Prod:output:0*integer_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2!
integer_lookup/bincount/Greater?
integer_lookup/bincount/CastCast#integer_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
integer_lookup/bincount/Cast?
integer_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
integer_lookup/bincount/Const_1?
integer_lookup/bincount/MaxMax integer_lookup/Identity:output:0(integer_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/Max?
integer_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
integer_lookup/bincount/add/y?
integer_lookup/bincount/addAddV2$integer_lookup/bincount/Max:output:0&integer_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/add?
integer_lookup/bincount/mulMul integer_lookup/bincount/Cast:y:0integer_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/mul?
!integer_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!integer_lookup/bincount/minlength?
integer_lookup/bincount/MaximumMaximum*integer_lookup/bincount/minlength:output:0integer_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2!
integer_lookup/bincount/Maximum?
!integer_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!integer_lookup/bincount/maxlength?
integer_lookup/bincount/MinimumMinimum*integer_lookup/bincount/maxlength:output:0#integer_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2!
integer_lookup/bincount/Minimum?
integer_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2!
integer_lookup/bincount/Const_2?
%integer_lookup/bincount/DenseBincountDenseBincount integer_lookup/Identity:output:0#integer_lookup/bincount/Minimum:z:0(integer_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2'
%integer_lookup/bincount/DenseBincount?
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleflat_model_type?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????23
1string_lookup/hash_table_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup/Identity?
string_lookup/bincount/ShapeShapestring_lookup/Identity:output:0*
T0	*
_output_shapes
:2
string_lookup/bincount/Shape?
string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
string_lookup/bincount/Const?
string_lookup/bincount/ProdProd%string_lookup/bincount/Shape:output:0%string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2
string_lookup/bincount/Prod?
 string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 string_lookup/bincount/Greater/y?
string_lookup/bincount/GreaterGreater$string_lookup/bincount/Prod:output:0)string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2 
string_lookup/bincount/Greater?
string_lookup/bincount/CastCast"string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
string_lookup/bincount/Cast?
string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
string_lookup/bincount/Const_1?
string_lookup/bincount/MaxMaxstring_lookup/Identity:output:0'string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/Max~
string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
string_lookup/bincount/add/y?
string_lookup/bincount/addAddV2#string_lookup/bincount/Max:output:0%string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/add?
string_lookup/bincount/mulMulstring_lookup/bincount/Cast:y:0string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/mul?
 string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2"
 string_lookup/bincount/minlength?
string_lookup/bincount/MaximumMaximum)string_lookup/bincount/minlength:output:0string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2 
string_lookup/bincount/Maximum?
 string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2"
 string_lookup/bincount/maxlength?
string_lookup/bincount/MinimumMinimum)string_lookup/bincount/maxlength:output:0"string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2 
string_lookup/bincount/Minimum?
string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2 
string_lookup/bincount/Const_2?
$string_lookup/bincount/DenseBincountDenseBincountstring_lookup/Identity:output:0"string_lookup/bincount/Minimum:z:0'string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2&
$string_lookup/bincount/DenseBincount?
3string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handlestorey_rangeAstring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity<string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_1/Identity?
string_lookup_1/bincount/ShapeShape!string_lookup_1/Identity:output:0*
T0	*
_output_shapes
:2 
string_lookup_1/bincount/Shape?
string_lookup_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
string_lookup_1/bincount/Const?
string_lookup_1/bincount/ProdProd'string_lookup_1/bincount/Shape:output:0'string_lookup_1/bincount/Const:output:0*
T0*
_output_shapes
: 2
string_lookup_1/bincount/Prod?
"string_lookup_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"string_lookup_1/bincount/Greater/y?
 string_lookup_1/bincount/GreaterGreater&string_lookup_1/bincount/Prod:output:0+string_lookup_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2"
 string_lookup_1/bincount/Greater?
string_lookup_1/bincount/CastCast$string_lookup_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
string_lookup_1/bincount/Cast?
 string_lookup_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 string_lookup_1/bincount/Const_1?
string_lookup_1/bincount/MaxMax!string_lookup_1/Identity:output:0)string_lookup_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/Max?
string_lookup_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2 
string_lookup_1/bincount/add/y?
string_lookup_1/bincount/addAddV2%string_lookup_1/bincount/Max:output:0'string_lookup_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/add?
string_lookup_1/bincount/mulMul!string_lookup_1/bincount/Cast:y:0 string_lookup_1/bincount/add:z:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/mul?
"string_lookup_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"string_lookup_1/bincount/minlength?
 string_lookup_1/bincount/MaximumMaximum+string_lookup_1/bincount/minlength:output:0 string_lookup_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2"
 string_lookup_1/bincount/Maximum?
"string_lookup_1/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"string_lookup_1/bincount/maxlength?
 string_lookup_1/bincount/MinimumMinimum+string_lookup_1/bincount/maxlength:output:0$string_lookup_1/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2"
 string_lookup_1/bincount/Minimum?
 string_lookup_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2"
 string_lookup_1/bincount/Const_2?
&string_lookup_1/bincount/DenseBincountDenseBincount!string_lookup_1/Identity:output:0$string_lookup_1/bincount/Minimum:z:0)string_lookup_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2(
&string_lookup_1/bincount/DenseBincount?
normalization/subSubfloor_area_sqmnormalization_sub_y*
T0*'
_output_shapes
:?????????2
normalization/subo
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
normalization_1/subSubdist_to_nearest_stnnormalization_1_sub_y*
T0*'
_output_shapes
:?????????2
normalization_1/subu
normalization_1/SqrtSqrtnormalization_1_sqrt_x*
T0*
_output_shapes

:2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_1/Maximum/y?
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_1/Maximum?
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_1/truediv?
normalization_2/subSubdist_to_dhobynormalization_2_sub_y*
T0*'
_output_shapes
:?????????2
normalization_2/subu
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes

:2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_2/Maximum/y?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_2/Maximum?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_2/truediv?
normalization_3/subSubdegree_centralitynormalization_3_sub_y*
T0*'
_output_shapes
:?????????2
normalization_3/subu
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_3/truediv?
normalization_4/subSubeigenvector_centralitynormalization_4_sub_y*
T0*'
_output_shapes
:?????????2
normalization_4/subu
normalization_4/SqrtSqrtnormalization_4_sqrt_x*
T0*
_output_shapes

:2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_4/Maximum/y?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_4/Maximum?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_4/truediv?
normalization_5/subSubremaining_lease_yearsnormalization_5_sub_y*
T0*'
_output_shapes
:?????????2
normalization_5/subu
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes

:2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_5/Maximum/y?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_5/Maximum?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_5/truediv?
concatenate/PartitionedCallPartitionedCall.integer_lookup/bincount/DenseBincount:output:0-string_lookup/bincount/DenseBincount:output:0/string_lookup_1/bincount/DenseBincount:output:0normalization/truediv:z:0normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Q* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2938712
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_294541dense_294543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2938842
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_294546dense_1_294548*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2939002!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall3^integer_lookup/hash_table_Lookup/LookupTableFindV22^string_lookup/hash_table_Lookup/LookupTableFindV24^string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2h
2integer_lookup/hash_table_Lookup/LookupTableFindV22integer_lookup/hash_table_Lookup/LookupTableFindV22f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV22j
3string_lookup_1/hash_table_Lookup/LookupTableFindV23string_lookup_1/hash_table_Lookup/LookupTableFindV2:N J
'
_output_shapes
:?????????

_user_specified_namemonth:XT
'
_output_shapes
:?????????
)
_user_specified_nameflat_model_type:UQ
'
_output_shapes
:?????????
&
_user_specified_namestorey_range:WS
'
_output_shapes
:?????????
(
_user_specified_namefloor_area_sqm:\X
'
_output_shapes
:?????????
-
_user_specified_namedist_to_nearest_stn:VR
'
_output_shapes
:?????????
'
_user_specified_namedist_to_dhoby:ZV
'
_output_shapes
:?????????
+
_user_specified_namedegree_centrality:_[
'
_output_shapes
:?????????
0
_user_specified_nameeigenvector_centrality:^Z
'
_output_shapes
:?????????
/
_user_specified_nameremaining_lease_years:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
-
__inference__destroyer_295151
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
G
__inference__creator_295156
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_284*
value_dtype0	2
MutableHashTablei
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identitya
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
G
__inference__creator_295090
identity:	 ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name
table_64*
value_dtype0	2
MutableHashTablei
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identitya
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
+
__inference_<lambda>_295273
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?,
?
__inference_adapt_step_7926
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNexts
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
?
__inference_<lambda>_2952686
2key_value_init250_lookuptableimportv2_table_handle.
*key_value_init250_lookuptableimportv2_keys0
,key_value_init250_lookuptableimportv2_values	
identity??%key_value_init250/LookupTableImportV2?
%key_value_init250/LookupTableImportV2LookupTableImportV22key_value_init250_lookuptableimportv2_table_handle*key_value_init250_lookuptableimportv2_keys,key_value_init250_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2'
%key_value_init250/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init250/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :+:+2N
%key_value_init250/LookupTableImportV2%key_value_init250/LookupTableImportV2: 

_output_shapes
:+: 

_output_shapes
:+
?
+
__inference_<lambda>_295260
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
&__inference_model_layer_call_fn_295001
inputs_0	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17:Q


unknown_18:


unknown_19:


unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20**
Tin#
!2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2941862
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
/
__inference__initializer_295161
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
-
__inference__destroyer_295085
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
;
__inference__creator_295105
identity??
hash_tabley

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name251*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
$__inference_signature_wrapper_294617
degree_centrality
dist_to_dhoby
dist_to_nearest_stn
eigenvector_centrality
flat_model_type
floor_area_sqm	
month	
remaining_lease_years
storey_range
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17:Q


unknown_18:


unknown_19:


unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmonthflat_model_typestorey_rangefloor_area_sqmdist_to_nearest_stndist_to_dhobydegree_centralityeigenvector_centralityremaining_lease_yearsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20**
Tin#
!2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_2937262
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:?????????
+
_user_specified_namedegree_centrality:VR
'
_output_shapes
:?????????
'
_user_specified_namedist_to_dhoby:\X
'
_output_shapes
:?????????
-
_user_specified_namedist_to_nearest_stn:_[
'
_output_shapes
:?????????
0
_user_specified_nameeigenvector_centrality:XT
'
_output_shapes
:?????????
)
_user_specified_nameflat_model_type:WS
'
_output_shapes
:?????????
(
_user_specified_namefloor_area_sqm:NJ
'
_output_shapes
:?????????

_user_specified_namemonth:^Z
'
_output_shapes
:?????????
/
_user_specified_nameremaining_lease_years:UQ
'
_output_shapes
:?????????
&
_user_specified_namestorey_range:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
-
__inference__destroyer_295133
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference_save_fn_295185
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2	

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0	*
Tvalues0	*
_output_shapes

::2A
?MutableHashTable_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1Q
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const\

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0	*
_output_shapes
:2

Identity_2W

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1^

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:2

Identity_5?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
-
__inference__destroyer_295166
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?	
?
__inference_restore_fn_295220
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 24
2MutableHashTable_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_293900

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
&__inference_dense_layer_call_fn_295048

inputs
unknown:Q

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2938842
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Q: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????Q
 
_user_specified_nameinputs
?
?
__inference_save_fn_295212
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2A
?MutableHashTable_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1Q
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const\

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:2

Identity_2W

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1^

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:2

Identity_5?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
__inference_<lambda>_2952556
2key_value_init140_lookuptableimportv2_table_handle.
*key_value_init140_lookuptableimportv2_keys	0
,key_value_init140_lookuptableimportv2_values	
identity??%key_value_init140/LookupTableImportV2?
%key_value_init140/LookupTableImportV2LookupTableImportV22key_value_init140_lookuptableimportv2_table_handle*key_value_init140_lookuptableimportv2_keys,key_value_init140_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 2'
%key_value_init140/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init140/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2N
%key_value_init140/LookupTableImportV2%key_value_init140/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
;
__inference__creator_295138
identity??
hash_tabley

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name361*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
??
?
!__inference__wrapped_model_293726	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_yearsI
Emodel_integer_lookup_hash_table_lookup_lookuptablefindv2_table_handleJ
Fmodel_integer_lookup_hash_table_lookup_lookuptablefindv2_default_value	H
Dmodel_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleI
Emodel_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	J
Fmodel_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleK
Gmodel_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	
model_normalization_sub_y
model_normalization_sqrt_x
model_normalization_1_sub_y 
model_normalization_1_sqrt_x
model_normalization_2_sub_y 
model_normalization_2_sqrt_x
model_normalization_3_sub_y 
model_normalization_3_sqrt_x
model_normalization_4_sub_y 
model_normalization_4_sqrt_x
model_normalization_5_sub_y 
model_normalization_5_sqrt_x<
*model_dense_matmul_readvariableop_resource:Q
9
+model_dense_biasadd_readvariableop_resource:
>
,model_dense_1_matmul_readvariableop_resource:
;
-model_dense_1_biasadd_readvariableop_resource:
identity??"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?8model/integer_lookup/hash_table_Lookup/LookupTableFindV2?7model/string_lookup/hash_table_Lookup/LookupTableFindV2?9model/string_lookup_1/hash_table_Lookup/LookupTableFindV2?
8model/integer_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Emodel_integer_lookup_hash_table_lookup_lookuptablefindv2_table_handlemonthFmodel_integer_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????2:
8model/integer_lookup/hash_table_Lookup/LookupTableFindV2?
model/integer_lookup/IdentityIdentityAmodel/integer_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
model/integer_lookup/Identity?
#model/integer_lookup/bincount/ShapeShape&model/integer_lookup/Identity:output:0*
T0	*
_output_shapes
:2%
#model/integer_lookup/bincount/Shape?
#model/integer_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model/integer_lookup/bincount/Const?
"model/integer_lookup/bincount/ProdProd,model/integer_lookup/bincount/Shape:output:0,model/integer_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2$
"model/integer_lookup/bincount/Prod?
'model/integer_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model/integer_lookup/bincount/Greater/y?
%model/integer_lookup/bincount/GreaterGreater+model/integer_lookup/bincount/Prod:output:00model/integer_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2'
%model/integer_lookup/bincount/Greater?
"model/integer_lookup/bincount/CastCast)model/integer_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2$
"model/integer_lookup/bincount/Cast?
%model/integer_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%model/integer_lookup/bincount/Const_1?
!model/integer_lookup/bincount/MaxMax&model/integer_lookup/Identity:output:0.model/integer_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2#
!model/integer_lookup/bincount/Max?
#model/integer_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2%
#model/integer_lookup/bincount/add/y?
!model/integer_lookup/bincount/addAddV2*model/integer_lookup/bincount/Max:output:0,model/integer_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2#
!model/integer_lookup/bincount/add?
!model/integer_lookup/bincount/mulMul&model/integer_lookup/bincount/Cast:y:0%model/integer_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2#
!model/integer_lookup/bincount/mul?
'model/integer_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2)
'model/integer_lookup/bincount/minlength?
%model/integer_lookup/bincount/MaximumMaximum0model/integer_lookup/bincount/minlength:output:0%model/integer_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2'
%model/integer_lookup/bincount/Maximum?
'model/integer_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2)
'model/integer_lookup/bincount/maxlength?
%model/integer_lookup/bincount/MinimumMinimum0model/integer_lookup/bincount/maxlength:output:0)model/integer_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2'
%model/integer_lookup/bincount/Minimum?
%model/integer_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2'
%model/integer_lookup/bincount/Const_2?
+model/integer_lookup/bincount/DenseBincountDenseBincount&model/integer_lookup/Identity:output:0)model/integer_lookup/bincount/Minimum:z:0.model/integer_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2-
+model/integer_lookup/bincount/DenseBincount?
7model/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Dmodel_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleflat_model_typeEmodel_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????29
7model/string_lookup/hash_table_Lookup/LookupTableFindV2?
model/string_lookup/IdentityIdentity@model/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
model/string_lookup/Identity?
"model/string_lookup/bincount/ShapeShape%model/string_lookup/Identity:output:0*
T0	*
_output_shapes
:2$
"model/string_lookup/bincount/Shape?
"model/string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"model/string_lookup/bincount/Const?
!model/string_lookup/bincount/ProdProd+model/string_lookup/bincount/Shape:output:0+model/string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!model/string_lookup/bincount/Prod?
&model/string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/string_lookup/bincount/Greater/y?
$model/string_lookup/bincount/GreaterGreater*model/string_lookup/bincount/Prod:output:0/model/string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$model/string_lookup/bincount/Greater?
!model/string_lookup/bincount/CastCast(model/string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!model/string_lookup/bincount/Cast?
$model/string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$model/string_lookup/bincount/Const_1?
 model/string_lookup/bincount/MaxMax%model/string_lookup/Identity:output:0-model/string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 model/string_lookup/bincount/Max?
"model/string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"model/string_lookup/bincount/add/y?
 model/string_lookup/bincount/addAddV2)model/string_lookup/bincount/Max:output:0+model/string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 model/string_lookup/bincount/add?
 model/string_lookup/bincount/mulMul%model/string_lookup/bincount/Cast:y:0$model/string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 model/string_lookup/bincount/mul?
&model/string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2(
&model/string_lookup/bincount/minlength?
$model/string_lookup/bincount/MaximumMaximum/model/string_lookup/bincount/minlength:output:0$model/string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$model/string_lookup/bincount/Maximum?
&model/string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2(
&model/string_lookup/bincount/maxlength?
$model/string_lookup/bincount/MinimumMinimum/model/string_lookup/bincount/maxlength:output:0(model/string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$model/string_lookup/bincount/Minimum?
$model/string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$model/string_lookup/bincount/Const_2?
*model/string_lookup/bincount/DenseBincountDenseBincount%model/string_lookup/Identity:output:0(model/string_lookup/bincount/Minimum:z:0-model/string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2,
*model/string_lookup/bincount/DenseBincount?
9model/string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Fmodel_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handlestorey_rangeGmodel_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2;
9model/string_lookup_1/hash_table_Lookup/LookupTableFindV2?
model/string_lookup_1/IdentityIdentityBmodel/string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_1/Identity?
$model/string_lookup_1/bincount/ShapeShape'model/string_lookup_1/Identity:output:0*
T0	*
_output_shapes
:2&
$model/string_lookup_1/bincount/Shape?
$model/string_lookup_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model/string_lookup_1/bincount/Const?
#model/string_lookup_1/bincount/ProdProd-model/string_lookup_1/bincount/Shape:output:0-model/string_lookup_1/bincount/Const:output:0*
T0*
_output_shapes
: 2%
#model/string_lookup_1/bincount/Prod?
(model/string_lookup_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model/string_lookup_1/bincount/Greater/y?
&model/string_lookup_1/bincount/GreaterGreater,model/string_lookup_1/bincount/Prod:output:01model/string_lookup_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2(
&model/string_lookup_1/bincount/Greater?
#model/string_lookup_1/bincount/CastCast*model/string_lookup_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2%
#model/string_lookup_1/bincount/Cast?
&model/string_lookup_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&model/string_lookup_1/bincount/Const_1?
"model/string_lookup_1/bincount/MaxMax'model/string_lookup_1/Identity:output:0/model/string_lookup_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2$
"model/string_lookup_1/bincount/Max?
$model/string_lookup_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2&
$model/string_lookup_1/bincount/add/y?
"model/string_lookup_1/bincount/addAddV2+model/string_lookup_1/bincount/Max:output:0-model/string_lookup_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2$
"model/string_lookup_1/bincount/add?
"model/string_lookup_1/bincount/mulMul'model/string_lookup_1/bincount/Cast:y:0&model/string_lookup_1/bincount/add:z:0*
T0	*
_output_shapes
: 2$
"model/string_lookup_1/bincount/mul?
(model/string_lookup_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/string_lookup_1/bincount/minlength?
&model/string_lookup_1/bincount/MaximumMaximum1model/string_lookup_1/bincount/minlength:output:0&model/string_lookup_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2(
&model/string_lookup_1/bincount/Maximum?
(model/string_lookup_1/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/string_lookup_1/bincount/maxlength?
&model/string_lookup_1/bincount/MinimumMinimum1model/string_lookup_1/bincount/maxlength:output:0*model/string_lookup_1/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2(
&model/string_lookup_1/bincount/Minimum?
&model/string_lookup_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&model/string_lookup_1/bincount/Const_2?
,model/string_lookup_1/bincount/DenseBincountDenseBincount'model/string_lookup_1/Identity:output:0*model/string_lookup_1/bincount/Minimum:z:0/model/string_lookup_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2.
,model/string_lookup_1/bincount/DenseBincount?
model/normalization/subSubfloor_area_sqmmodel_normalization_sub_y*
T0*'
_output_shapes
:?????????2
model/normalization/sub?
model/normalization/SqrtSqrtmodel_normalization_sqrt_x*
T0*
_output_shapes

:2
model/normalization/Sqrt?
model/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
model/normalization/Maximum/y?
model/normalization/MaximumMaximummodel/normalization/Sqrt:y:0&model/normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization/Maximum?
model/normalization/truedivRealDivmodel/normalization/sub:z:0model/normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
model/normalization/truediv?
model/normalization_1/subSubdist_to_nearest_stnmodel_normalization_1_sub_y*
T0*'
_output_shapes
:?????????2
model/normalization_1/sub?
model/normalization_1/SqrtSqrtmodel_normalization_1_sqrt_x*
T0*
_output_shapes

:2
model/normalization_1/Sqrt?
model/normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32!
model/normalization_1/Maximum/y?
model/normalization_1/MaximumMaximummodel/normalization_1/Sqrt:y:0(model/normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization_1/Maximum?
model/normalization_1/truedivRealDivmodel/normalization_1/sub:z:0!model/normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????2
model/normalization_1/truediv?
model/normalization_2/subSubdist_to_dhobymodel_normalization_2_sub_y*
T0*'
_output_shapes
:?????????2
model/normalization_2/sub?
model/normalization_2/SqrtSqrtmodel_normalization_2_sqrt_x*
T0*
_output_shapes

:2
model/normalization_2/Sqrt?
model/normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32!
model/normalization_2/Maximum/y?
model/normalization_2/MaximumMaximummodel/normalization_2/Sqrt:y:0(model/normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization_2/Maximum?
model/normalization_2/truedivRealDivmodel/normalization_2/sub:z:0!model/normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????2
model/normalization_2/truediv?
model/normalization_3/subSubdegree_centralitymodel_normalization_3_sub_y*
T0*'
_output_shapes
:?????????2
model/normalization_3/sub?
model/normalization_3/SqrtSqrtmodel_normalization_3_sqrt_x*
T0*
_output_shapes

:2
model/normalization_3/Sqrt?
model/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32!
model/normalization_3/Maximum/y?
model/normalization_3/MaximumMaximummodel/normalization_3/Sqrt:y:0(model/normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization_3/Maximum?
model/normalization_3/truedivRealDivmodel/normalization_3/sub:z:0!model/normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
model/normalization_3/truediv?
model/normalization_4/subSubeigenvector_centralitymodel_normalization_4_sub_y*
T0*'
_output_shapes
:?????????2
model/normalization_4/sub?
model/normalization_4/SqrtSqrtmodel_normalization_4_sqrt_x*
T0*
_output_shapes

:2
model/normalization_4/Sqrt?
model/normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32!
model/normalization_4/Maximum/y?
model/normalization_4/MaximumMaximummodel/normalization_4/Sqrt:y:0(model/normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization_4/Maximum?
model/normalization_4/truedivRealDivmodel/normalization_4/sub:z:0!model/normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????2
model/normalization_4/truediv?
model/normalization_5/subSubremaining_lease_yearsmodel_normalization_5_sub_y*
T0*'
_output_shapes
:?????????2
model/normalization_5/sub?
model/normalization_5/SqrtSqrtmodel_normalization_5_sqrt_x*
T0*
_output_shapes

:2
model/normalization_5/Sqrt?
model/normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32!
model/normalization_5/Maximum/y?
model/normalization_5/MaximumMaximummodel/normalization_5/Sqrt:y:0(model/normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization_5/Maximum?
model/normalization_5/truedivRealDivmodel/normalization_5/sub:z:0!model/normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????2
model/normalization_5/truediv?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV24model/integer_lookup/bincount/DenseBincount:output:03model/string_lookup/bincount/DenseBincount:output:05model/string_lookup_1/bincount/DenseBincount:output:0model/normalization/truediv:z:0!model/normalization_1/truediv:z:0!model/normalization_2/truediv:z:0!model/normalization_3/truediv:z:0!model/normalization_4/truediv:z:0!model/normalization_5/truediv:z:0&model/concatenate/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????Q2
model/concatenate/concat?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:Q
*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
model/dense/Relu?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02%
#model/dense_1/MatMul/ReadVariableOp?
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_1/MatMul?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_1/BiasAddy
IdentityIdentitymodel/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp9^model/integer_lookup/hash_table_Lookup/LookupTableFindV28^model/string_lookup/hash_table_Lookup/LookupTableFindV2:^model/string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2t
8model/integer_lookup/hash_table_Lookup/LookupTableFindV28model/integer_lookup/hash_table_Lookup/LookupTableFindV22r
7model/string_lookup/hash_table_Lookup/LookupTableFindV27model/string_lookup/hash_table_Lookup/LookupTableFindV22v
9model/string_lookup_1/hash_table_Lookup/LookupTableFindV29model/string_lookup_1/hash_table_Lookup/LookupTableFindV2:N J
'
_output_shapes
:?????????

_user_specified_namemonth:XT
'
_output_shapes
:?????????
)
_user_specified_nameflat_model_type:UQ
'
_output_shapes
:?????????
&
_user_specified_namestorey_range:WS
'
_output_shapes
:?????????
(
_user_specified_namefloor_area_sqm:\X
'
_output_shapes
:?????????
-
_user_specified_namedist_to_nearest_stn:VR
'
_output_shapes
:?????????
'
_user_specified_namedist_to_dhoby:ZV
'
_output_shapes
:?????????
+
_user_specified_namedegree_centrality:_[
'
_output_shapes
:?????????
0
_user_specified_nameeigenvector_centrality:^Z
'
_output_shapes
:?????????
/
_user_specified_nameremaining_lease_years:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?,
?
__inference_adapt_step_7785
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNexts
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
G
__inference__creator_295123
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_174*
value_dtype0	2
MutableHashTablei
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identitya
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?

?
A__inference_dense_layer_call_and_return_conditional_losses_293884

inputs0
matmul_readvariableop_resource:Q
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Q
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Q: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Q
 
_user_specified_nameinputs
?	
?
__inference_restore_fn_295247
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 24
2MutableHashTable_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
?
__inference_<lambda>_2952816
2key_value_init360_lookuptableimportv2_table_handle.
*key_value_init360_lookuptableimportv2_keys0
,key_value_init360_lookuptableimportv2_values	
identity??%key_value_init360/LookupTableImportV2?
%key_value_init360/LookupTableImportV2LookupTableImportV22key_value_init360_lookuptableimportv2_table_handle*key_value_init360_lookuptableimportv2_keys,key_value_init360_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2'
%key_value_init360/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init360/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2N
%key_value_init360/LookupTableImportV2%key_value_init360/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
G__inference_concatenate_layer_call_and_return_conditional_losses_295015
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????Q2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????,:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????,
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8
?
+
__inference_<lambda>_295286
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
/
__inference__initializer_295095
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference_adapt_step_7663
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
2	2
IteratorGetNextq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shape?
ReshapeReshapeIteratorGetNext:components:0Reshape/shape:output:0*
T0	*#
_output_shapes
:?????????2	
Reshape?
UniqueWithCountsUniqueWithCountsReshape:output:0*
T0	*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	2
UniqueWithCounts?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0	*

Tout0	*
_output_shapes
:2*
(None_lookup_table_find/LookupTableFindV2?
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2
add?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0	*

Tout0	*
_output_shapes
 2.
,None_lookup_table_insert/LookupTableInsertV2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?
?
__inference__initializer_2950806
2key_value_init140_lookuptableimportv2_table_handle.
*key_value_init140_lookuptableimportv2_keys	0
,key_value_init140_lookuptableimportv2_values	
identity??%key_value_init140/LookupTableImportV2?
%key_value_init140/LookupTableImportV2LookupTableImportV22key_value_init140_lookuptableimportv2_table_handle*key_value_init140_lookuptableimportv2_keys,key_value_init140_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 2'
%key_value_init140/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init140/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2N
%key_value_init140/LookupTableImportV2%key_value_init140/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference__initializer_2951136
2key_value_init250_lookuptableimportv2_table_handle.
*key_value_init250_lookuptableimportv2_keys0
,key_value_init250_lookuptableimportv2_values	
identity??%key_value_init250/LookupTableImportV2?
%key_value_init250/LookupTableImportV2LookupTableImportV22key_value_init250_lookuptableimportv2_table_handle*key_value_init250_lookuptableimportv2_keys,key_value_init250_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2'
%key_value_init250/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init250/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :+:+2N
%key_value_init250/LookupTableImportV2%key_value_init250/LookupTableImportV2: 

_output_shapes
:+: 

_output_shapes
:+
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_295058

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_294944
inputs_0	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17:Q


unknown_18:


unknown_19:


unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20**
Tin#
!2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2939072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
??
?
"__inference__traced_restore_295642
file_prefixM
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable:	 Q
Gmutablehashtable_table_restore_1_lookuptableimportv2_mutablehashtable_1: Q
Gmutablehashtable_table_restore_2_lookuptableimportv2_mutablehashtable_2: #
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 '
assignvariableop_3_mean_1:+
assignvariableop_4_variance_1:$
assignvariableop_5_count_1:	 '
assignvariableop_6_mean_2:+
assignvariableop_7_variance_2:$
assignvariableop_8_count_2:	 '
assignvariableop_9_mean_3:,
assignvariableop_10_variance_3:%
assignvariableop_11_count_3:	 (
assignvariableop_12_mean_4:,
assignvariableop_13_variance_4:%
assignvariableop_14_count_4:	 (
assignvariableop_15_mean_5:,
assignvariableop_16_variance_5:%
assignvariableop_17_count_5:	 2
 assignvariableop_18_dense_kernel:Q
,
assignvariableop_19_dense_bias:
4
"assignvariableop_20_dense_1_kernel:
.
 assignvariableop_21_dense_1_bias:'
assignvariableop_22_adam_iter:	 )
assignvariableop_23_adam_beta_1: )
assignvariableop_24_adam_beta_2: (
assignvariableop_25_adam_decay: 0
&assignvariableop_26_adam_learning_rate: #
assignvariableop_27_total: %
assignvariableop_28_count_6: %
assignvariableop_29_total_1: %
assignvariableop_30_count_7: %
assignvariableop_31_total_2: %
assignvariableop_32_count_8: %
assignvariableop_33_total_3: %
assignvariableop_34_count_9: 9
'assignvariableop_35_adam_dense_kernel_m:Q
3
%assignvariableop_36_adam_dense_bias_m:
;
)assignvariableop_37_adam_dense_1_kernel_m:
5
'assignvariableop_38_adam_dense_1_bias_m:9
'assignvariableop_39_adam_dense_kernel_v:Q
3
%assignvariableop_40_adam_dense_bias_v:
;
)assignvariableop_41_adam_dense_1_kernel_v:
5
'assignvariableop_42_adam_dense_1_bias_v:
identity_44??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?4MutableHashTable_table_restore_1/LookupTableImportV2?4MutableHashTable_table_restore_2/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B8layer_with_weights-0/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-0/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-1/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-1/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-2/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-2/token_counts/.ATTRIBUTES/table-valuesB4layer_with_weights-3/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-3/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-7/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-8/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422											2
	RestoreV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:0RestoreV2:tensors:1*	
Tin0	*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 24
2MutableHashTable_table_restore/LookupTableImportV2?
4MutableHashTable_table_restore_1/LookupTableImportV2LookupTableImportV2Gmutablehashtable_table_restore_1_lookuptableimportv2_mutablehashtable_1RestoreV2:tensors:2RestoreV2:tensors:3*	
Tin0*

Tout0	*%
_class
loc:@MutableHashTable_1*
_output_shapes
 26
4MutableHashTable_table_restore_1/LookupTableImportV2?
4MutableHashTable_table_restore_2/LookupTableImportV2LookupTableImportV2Gmutablehashtable_table_restore_2_lookuptableimportv2_mutablehashtable_2RestoreV2:tensors:4RestoreV2:tensors:5*	
Tin0*

Tout0	*%
_class
loc:@MutableHashTable_2*
_output_shapes
 26
4MutableHashTable_table_restore_2/LookupTableImportV2g
IdentityIdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_mean_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3l

Identity_4IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_variance_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4l

Identity_5IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_count_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5l

Identity_6IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_mean_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6l

Identity_7IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_variance_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7l

Identity_8IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_count_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8l

Identity_9IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_mean_3Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_variance_3Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_3Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_mean_4Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_variance_4Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_4Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_mean_5Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_variance_5Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_5Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_dense_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_1_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_1_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_6Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_7Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_2Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_8Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_3Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_9Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_dense_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_dense_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV25^MutableHashTable_table_restore_1/LookupTableImportV25^MutableHashTable_table_restore_2/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43f
Identity_44IdentityIdentity_43:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_44?	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV25^MutableHashTable_table_restore_1/LookupTableImportV25^MutableHashTable_table_restore_2/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_44Identity_44:output:0*q
_input_shapes`
^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV22l
4MutableHashTable_table_restore_1/LookupTableImportV24MutableHashTable_table_restore_1/LookupTableImportV22l
4MutableHashTable_table_restore_2/LookupTableImportV24MutableHashTable_table_restore_2/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable:+'
%
_class
loc:@MutableHashTable_1:+'
%
_class
loc:@MutableHashTable_2
?,
?
__inference_adapt_step_7973
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNexts
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
?
,__inference_concatenate_layer_call_fn_295028
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Q* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2938712
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????,:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????,
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8
?
;
__inference__creator_295072
identity??
hash_tabley

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name141*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference_save_fn_295239
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2A
?MutableHashTable_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1Q
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const\

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:2

Identity_2W

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1^

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:2

Identity_5?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
-
__inference__destroyer_295118
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?,
?
__inference_adapt_step_7832
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNexts
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
ɠ
?
A__inference_model_layer_call_and_return_conditional_losses_294887
inputs_0	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8C
?integer_lookup_hash_table_lookup_lookuptablefindv2_table_handleD
@integer_lookup_hash_table_lookup_lookuptablefindv2_default_value	B
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	D
@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleE
Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	
normalization_sub_y
normalization_sqrt_x
normalization_1_sub_y
normalization_1_sqrt_x
normalization_2_sub_y
normalization_2_sqrt_x
normalization_3_sub_y
normalization_3_sqrt_x
normalization_4_sub_y
normalization_4_sqrt_x
normalization_5_sub_y
normalization_5_sqrt_x6
$dense_matmul_readvariableop_resource:Q
3
%dense_biasadd_readvariableop_resource:
8
&dense_1_matmul_readvariableop_resource:
5
'dense_1_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?2integer_lookup/hash_table_Lookup/LookupTableFindV2?1string_lookup/hash_table_Lookup/LookupTableFindV2?3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
2integer_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2?integer_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs_0@integer_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????24
2integer_lookup/hash_table_Lookup/LookupTableFindV2?
integer_lookup/IdentityIdentity;integer_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup/Identity?
integer_lookup/bincount/ShapeShape integer_lookup/Identity:output:0*
T0	*
_output_shapes
:2
integer_lookup/bincount/Shape?
integer_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
integer_lookup/bincount/Const?
integer_lookup/bincount/ProdProd&integer_lookup/bincount/Shape:output:0&integer_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2
integer_lookup/bincount/Prod?
!integer_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!integer_lookup/bincount/Greater/y?
integer_lookup/bincount/GreaterGreater%integer_lookup/bincount/Prod:output:0*integer_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2!
integer_lookup/bincount/Greater?
integer_lookup/bincount/CastCast#integer_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
integer_lookup/bincount/Cast?
integer_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
integer_lookup/bincount/Const_1?
integer_lookup/bincount/MaxMax integer_lookup/Identity:output:0(integer_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/Max?
integer_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
integer_lookup/bincount/add/y?
integer_lookup/bincount/addAddV2$integer_lookup/bincount/Max:output:0&integer_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/add?
integer_lookup/bincount/mulMul integer_lookup/bincount/Cast:y:0integer_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/mul?
!integer_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!integer_lookup/bincount/minlength?
integer_lookup/bincount/MaximumMaximum*integer_lookup/bincount/minlength:output:0integer_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2!
integer_lookup/bincount/Maximum?
!integer_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!integer_lookup/bincount/maxlength?
integer_lookup/bincount/MinimumMinimum*integer_lookup/bincount/maxlength:output:0#integer_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2!
integer_lookup/bincount/Minimum?
integer_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2!
integer_lookup/bincount/Const_2?
%integer_lookup/bincount/DenseBincountDenseBincount integer_lookup/Identity:output:0#integer_lookup/bincount/Minimum:z:0(integer_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2'
%integer_lookup/bincount/DenseBincount?
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs_1?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????23
1string_lookup/hash_table_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup/Identity?
string_lookup/bincount/ShapeShapestring_lookup/Identity:output:0*
T0	*
_output_shapes
:2
string_lookup/bincount/Shape?
string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
string_lookup/bincount/Const?
string_lookup/bincount/ProdProd%string_lookup/bincount/Shape:output:0%string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2
string_lookup/bincount/Prod?
 string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 string_lookup/bincount/Greater/y?
string_lookup/bincount/GreaterGreater$string_lookup/bincount/Prod:output:0)string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2 
string_lookup/bincount/Greater?
string_lookup/bincount/CastCast"string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
string_lookup/bincount/Cast?
string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
string_lookup/bincount/Const_1?
string_lookup/bincount/MaxMaxstring_lookup/Identity:output:0'string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/Max~
string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
string_lookup/bincount/add/y?
string_lookup/bincount/addAddV2#string_lookup/bincount/Max:output:0%string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/add?
string_lookup/bincount/mulMulstring_lookup/bincount/Cast:y:0string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/mul?
 string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2"
 string_lookup/bincount/minlength?
string_lookup/bincount/MaximumMaximum)string_lookup/bincount/minlength:output:0string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2 
string_lookup/bincount/Maximum?
 string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2"
 string_lookup/bincount/maxlength?
string_lookup/bincount/MinimumMinimum)string_lookup/bincount/maxlength:output:0"string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2 
string_lookup/bincount/Minimum?
string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2 
string_lookup/bincount/Const_2?
$string_lookup/bincount/DenseBincountDenseBincountstring_lookup/Identity:output:0"string_lookup/bincount/Minimum:z:0'string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2&
$string_lookup/bincount/DenseBincount?
3string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleinputs_2Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity<string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_1/Identity?
string_lookup_1/bincount/ShapeShape!string_lookup_1/Identity:output:0*
T0	*
_output_shapes
:2 
string_lookup_1/bincount/Shape?
string_lookup_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
string_lookup_1/bincount/Const?
string_lookup_1/bincount/ProdProd'string_lookup_1/bincount/Shape:output:0'string_lookup_1/bincount/Const:output:0*
T0*
_output_shapes
: 2
string_lookup_1/bincount/Prod?
"string_lookup_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"string_lookup_1/bincount/Greater/y?
 string_lookup_1/bincount/GreaterGreater&string_lookup_1/bincount/Prod:output:0+string_lookup_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2"
 string_lookup_1/bincount/Greater?
string_lookup_1/bincount/CastCast$string_lookup_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
string_lookup_1/bincount/Cast?
 string_lookup_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 string_lookup_1/bincount/Const_1?
string_lookup_1/bincount/MaxMax!string_lookup_1/Identity:output:0)string_lookup_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/Max?
string_lookup_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2 
string_lookup_1/bincount/add/y?
string_lookup_1/bincount/addAddV2%string_lookup_1/bincount/Max:output:0'string_lookup_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/add?
string_lookup_1/bincount/mulMul!string_lookup_1/bincount/Cast:y:0 string_lookup_1/bincount/add:z:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/mul?
"string_lookup_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"string_lookup_1/bincount/minlength?
 string_lookup_1/bincount/MaximumMaximum+string_lookup_1/bincount/minlength:output:0 string_lookup_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2"
 string_lookup_1/bincount/Maximum?
"string_lookup_1/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"string_lookup_1/bincount/maxlength?
 string_lookup_1/bincount/MinimumMinimum+string_lookup_1/bincount/maxlength:output:0$string_lookup_1/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2"
 string_lookup_1/bincount/Minimum?
 string_lookup_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2"
 string_lookup_1/bincount/Const_2?
&string_lookup_1/bincount/DenseBincountDenseBincount!string_lookup_1/Identity:output:0$string_lookup_1/bincount/Minimum:z:0)string_lookup_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2(
&string_lookup_1/bincount/DenseBincount~
normalization/subSubinputs_3normalization_sub_y*
T0*'
_output_shapes
:?????????2
normalization/subo
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
normalization_1/subSubinputs_4normalization_1_sub_y*
T0*'
_output_shapes
:?????????2
normalization_1/subu
normalization_1/SqrtSqrtnormalization_1_sqrt_x*
T0*
_output_shapes

:2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_1/Maximum/y?
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_1/Maximum?
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_1/truediv?
normalization_2/subSubinputs_5normalization_2_sub_y*
T0*'
_output_shapes
:?????????2
normalization_2/subu
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes

:2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_2/Maximum/y?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_2/Maximum?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_2/truediv?
normalization_3/subSubinputs_6normalization_3_sub_y*
T0*'
_output_shapes
:?????????2
normalization_3/subu
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_3/truediv?
normalization_4/subSubinputs_7normalization_4_sub_y*
T0*'
_output_shapes
:?????????2
normalization_4/subu
normalization_4/SqrtSqrtnormalization_4_sqrt_x*
T0*
_output_shapes

:2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_4/Maximum/y?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_4/Maximum?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_4/truediv?
normalization_5/subSubinputs_8normalization_5_sub_y*
T0*'
_output_shapes
:?????????2
normalization_5/subu
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes

:2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_5/Maximum/y?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_5/Maximum?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_5/truedivt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2.integer_lookup/bincount/DenseBincount:output:0-string_lookup/bincount/DenseBincount:output:0/string_lookup_1/bincount/DenseBincount:output:0normalization/truediv:z:0normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0 concatenate/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????Q2
concatenate/concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:Q
*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdds
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp3^integer_lookup/hash_table_Lookup/LookupTableFindV22^string_lookup/hash_table_Lookup/LookupTableFindV24^string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2h
2integer_lookup/hash_table_Lookup/LookupTableFindV22integer_lookup/hash_table_Lookup/LookupTableFindV22f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV22j
3string_lookup_1/hash_table_Lookup/LookupTableFindV23string_lookup_1/hash_table_Lookup/LookupTableFindV2:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
G__inference_concatenate_layer_call_and_return_conditional_losses_293871

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????Q2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????,:?????????:?????????:?????????:?????????:?????????:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????,
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
__inference_restore_fn_295193
restored_tensors_0	
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0	*

Tout0	*
_output_shapes
 24
2MutableHashTable_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?,
?
__inference_adapt_step_7738
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNexts
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
-
__inference__destroyer_295100
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?

A__inference_model_layer_call_and_return_conditional_losses_293907

inputs	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8C
?integer_lookup_hash_table_lookup_lookuptablefindv2_table_handleD
@integer_lookup_hash_table_lookup_lookuptablefindv2_default_value	B
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	D
@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleE
Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	
normalization_sub_y
normalization_sqrt_x
normalization_1_sub_y
normalization_1_sqrt_x
normalization_2_sub_y
normalization_2_sqrt_x
normalization_3_sub_y
normalization_3_sqrt_x
normalization_4_sub_y
normalization_4_sqrt_x
normalization_5_sub_y
normalization_5_sqrt_x
dense_293885:Q

dense_293887:
 
dense_1_293901:

dense_1_293903:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?2integer_lookup/hash_table_Lookup/LookupTableFindV2?1string_lookup/hash_table_Lookup/LookupTableFindV2?3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
2integer_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2?integer_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs@integer_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????24
2integer_lookup/hash_table_Lookup/LookupTableFindV2?
integer_lookup/IdentityIdentity;integer_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup/Identity?
integer_lookup/bincount/ShapeShape integer_lookup/Identity:output:0*
T0	*
_output_shapes
:2
integer_lookup/bincount/Shape?
integer_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
integer_lookup/bincount/Const?
integer_lookup/bincount/ProdProd&integer_lookup/bincount/Shape:output:0&integer_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2
integer_lookup/bincount/Prod?
!integer_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!integer_lookup/bincount/Greater/y?
integer_lookup/bincount/GreaterGreater%integer_lookup/bincount/Prod:output:0*integer_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2!
integer_lookup/bincount/Greater?
integer_lookup/bincount/CastCast#integer_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
integer_lookup/bincount/Cast?
integer_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
integer_lookup/bincount/Const_1?
integer_lookup/bincount/MaxMax integer_lookup/Identity:output:0(integer_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/Max?
integer_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
integer_lookup/bincount/add/y?
integer_lookup/bincount/addAddV2$integer_lookup/bincount/Max:output:0&integer_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/add?
integer_lookup/bincount/mulMul integer_lookup/bincount/Cast:y:0integer_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/mul?
!integer_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!integer_lookup/bincount/minlength?
integer_lookup/bincount/MaximumMaximum*integer_lookup/bincount/minlength:output:0integer_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2!
integer_lookup/bincount/Maximum?
!integer_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!integer_lookup/bincount/maxlength?
integer_lookup/bincount/MinimumMinimum*integer_lookup/bincount/maxlength:output:0#integer_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2!
integer_lookup/bincount/Minimum?
integer_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2!
integer_lookup/bincount/Const_2?
%integer_lookup/bincount/DenseBincountDenseBincount integer_lookup/Identity:output:0#integer_lookup/bincount/Minimum:z:0(integer_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2'
%integer_lookup/bincount/DenseBincount?
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs_1?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????23
1string_lookup/hash_table_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup/Identity?
string_lookup/bincount/ShapeShapestring_lookup/Identity:output:0*
T0	*
_output_shapes
:2
string_lookup/bincount/Shape?
string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
string_lookup/bincount/Const?
string_lookup/bincount/ProdProd%string_lookup/bincount/Shape:output:0%string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2
string_lookup/bincount/Prod?
 string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 string_lookup/bincount/Greater/y?
string_lookup/bincount/GreaterGreater$string_lookup/bincount/Prod:output:0)string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2 
string_lookup/bincount/Greater?
string_lookup/bincount/CastCast"string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
string_lookup/bincount/Cast?
string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
string_lookup/bincount/Const_1?
string_lookup/bincount/MaxMaxstring_lookup/Identity:output:0'string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/Max~
string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
string_lookup/bincount/add/y?
string_lookup/bincount/addAddV2#string_lookup/bincount/Max:output:0%string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/add?
string_lookup/bincount/mulMulstring_lookup/bincount/Cast:y:0string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/mul?
 string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2"
 string_lookup/bincount/minlength?
string_lookup/bincount/MaximumMaximum)string_lookup/bincount/minlength:output:0string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2 
string_lookup/bincount/Maximum?
 string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2"
 string_lookup/bincount/maxlength?
string_lookup/bincount/MinimumMinimum)string_lookup/bincount/maxlength:output:0"string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2 
string_lookup/bincount/Minimum?
string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2 
string_lookup/bincount/Const_2?
$string_lookup/bincount/DenseBincountDenseBincountstring_lookup/Identity:output:0"string_lookup/bincount/Minimum:z:0'string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2&
$string_lookup/bincount/DenseBincount?
3string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleinputs_2Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity<string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_1/Identity?
string_lookup_1/bincount/ShapeShape!string_lookup_1/Identity:output:0*
T0	*
_output_shapes
:2 
string_lookup_1/bincount/Shape?
string_lookup_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
string_lookup_1/bincount/Const?
string_lookup_1/bincount/ProdProd'string_lookup_1/bincount/Shape:output:0'string_lookup_1/bincount/Const:output:0*
T0*
_output_shapes
: 2
string_lookup_1/bincount/Prod?
"string_lookup_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"string_lookup_1/bincount/Greater/y?
 string_lookup_1/bincount/GreaterGreater&string_lookup_1/bincount/Prod:output:0+string_lookup_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2"
 string_lookup_1/bincount/Greater?
string_lookup_1/bincount/CastCast$string_lookup_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
string_lookup_1/bincount/Cast?
 string_lookup_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 string_lookup_1/bincount/Const_1?
string_lookup_1/bincount/MaxMax!string_lookup_1/Identity:output:0)string_lookup_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/Max?
string_lookup_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2 
string_lookup_1/bincount/add/y?
string_lookup_1/bincount/addAddV2%string_lookup_1/bincount/Max:output:0'string_lookup_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/add?
string_lookup_1/bincount/mulMul!string_lookup_1/bincount/Cast:y:0 string_lookup_1/bincount/add:z:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/mul?
"string_lookup_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"string_lookup_1/bincount/minlength?
 string_lookup_1/bincount/MaximumMaximum+string_lookup_1/bincount/minlength:output:0 string_lookup_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2"
 string_lookup_1/bincount/Maximum?
"string_lookup_1/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"string_lookup_1/bincount/maxlength?
 string_lookup_1/bincount/MinimumMinimum+string_lookup_1/bincount/maxlength:output:0$string_lookup_1/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2"
 string_lookup_1/bincount/Minimum?
 string_lookup_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2"
 string_lookup_1/bincount/Const_2?
&string_lookup_1/bincount/DenseBincountDenseBincount!string_lookup_1/Identity:output:0$string_lookup_1/bincount/Minimum:z:0)string_lookup_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2(
&string_lookup_1/bincount/DenseBincount~
normalization/subSubinputs_3normalization_sub_y*
T0*'
_output_shapes
:?????????2
normalization/subo
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
normalization_1/subSubinputs_4normalization_1_sub_y*
T0*'
_output_shapes
:?????????2
normalization_1/subu
normalization_1/SqrtSqrtnormalization_1_sqrt_x*
T0*
_output_shapes

:2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_1/Maximum/y?
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_1/Maximum?
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_1/truediv?
normalization_2/subSubinputs_5normalization_2_sub_y*
T0*'
_output_shapes
:?????????2
normalization_2/subu
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes

:2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_2/Maximum/y?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_2/Maximum?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_2/truediv?
normalization_3/subSubinputs_6normalization_3_sub_y*
T0*'
_output_shapes
:?????????2
normalization_3/subu
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_3/truediv?
normalization_4/subSubinputs_7normalization_4_sub_y*
T0*'
_output_shapes
:?????????2
normalization_4/subu
normalization_4/SqrtSqrtnormalization_4_sqrt_x*
T0*
_output_shapes

:2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_4/Maximum/y?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_4/Maximum?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_4/truediv?
normalization_5/subSubinputs_8normalization_5_sub_y*
T0*'
_output_shapes
:?????????2
normalization_5/subu
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes

:2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_5/Maximum/y?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_5/Maximum?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_5/truediv?
concatenate/PartitionedCallPartitionedCall.integer_lookup/bincount/DenseBincount:output:0-string_lookup/bincount/DenseBincount:output:0/string_lookup_1/bincount/DenseBincount:output:0normalization/truediv:z:0normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Q* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2938712
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_293885dense_293887*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2938842
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_293901dense_1_293903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2939002!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall3^integer_lookup/hash_table_Lookup/LookupTableFindV22^string_lookup/hash_table_Lookup/LookupTableFindV24^string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2h
2integer_lookup/hash_table_Lookup/LookupTableFindV22integer_lookup/hash_table_Lookup/LookupTableFindV22f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV22j
3string_lookup_1/hash_table_Lookup/LookupTableFindV23string_lookup_1/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?,
?
__inference_adapt_step_7879
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNexts
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
?
__inference__initializer_2951466
2key_value_init360_lookuptableimportv2_table_handle.
*key_value_init360_lookuptableimportv2_keys0
,key_value_init360_lookuptableimportv2_values	
identity??%key_value_init360/LookupTableImportV2?
%key_value_init360/LookupTableImportV2LookupTableImportV22key_value_init360_lookuptableimportv2_table_handle*key_value_init360_lookuptableimportv2_keys,key_value_init360_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2'
%key_value_init360/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init360/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2N
%key_value_init360/LookupTableImportV2%key_value_init360/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?\
?
__inference__traced_save_295494
file_prefixJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2	L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	L
Hsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2N
Jsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2_1	L
Hsavev2_mutablehashtable_2_lookup_table_export_values_lookuptableexportv2N
Jsavev2_mutablehashtable_2_lookup_table_export_values_lookuptableexportv2_1	#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	%
!savev2_mean_1_read_readvariableop)
%savev2_variance_1_read_readvariableop&
"savev2_count_1_read_readvariableop	%
!savev2_mean_2_read_readvariableop)
%savev2_variance_2_read_readvariableop&
"savev2_count_2_read_readvariableop	%
!savev2_mean_3_read_readvariableop)
%savev2_variance_3_read_readvariableop&
"savev2_count_3_read_readvariableop	%
!savev2_mean_4_read_readvariableop)
%savev2_variance_4_read_readvariableop&
"savev2_count_4_read_readvariableop	%
!savev2_mean_5_read_readvariableop)
%savev2_variance_5_read_readvariableop&
"savev2_count_5_read_readvariableop	+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_6_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_7_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_8_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_9_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const_24

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B8layer_with_weights-0/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-0/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-1/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-1/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-2/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-2/token_counts/.ATTRIBUTES/table-valuesB4layer_with_weights-3/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-3/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-7/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-8/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1Hsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2Jsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2_1Hsavev2_mutablehashtable_2_lookup_table_export_values_lookuptableexportv2Jsavev2_mutablehashtable_2_lookup_table_export_values_lookuptableexportv2_1savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop!savev2_mean_1_read_readvariableop%savev2_variance_1_read_readvariableop"savev2_count_1_read_readvariableop!savev2_mean_2_read_readvariableop%savev2_variance_2_read_readvariableop"savev2_count_2_read_readvariableop!savev2_mean_3_read_readvariableop%savev2_variance_3_read_readvariableop"savev2_count_3_read_readvariableop!savev2_mean_4_read_readvariableop%savev2_variance_4_read_readvariableop"savev2_count_4_read_readvariableop!savev2_mean_5_read_readvariableop%savev2_variance_5_read_readvariableop"savev2_count_5_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop"savev2_count_6_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_7_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_8_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_9_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const_24"/device:CPU:0*
_output_shapes
 *@
dtypes6
422											2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::: ::: ::: ::: ::: ::: :Q
:
:
:: : : : : : : : : : : : : :Q
:
:
::Q
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::	

_output_shapes
: : 


_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :$ 

_output_shapes

:Q
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :$* 

_output_shapes

:Q
: +

_output_shapes
:
:$, 

_output_shapes

:
: -

_output_shapes
::$. 

_output_shapes

:Q
: /

_output_shapes
:
:$0 

_output_shapes

:
: 1

_output_shapes
::2

_output_shapes
: 
?
/
__inference__initializer_295128
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?

A__inference_model_layer_call_and_return_conditional_losses_294421	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_yearsC
?integer_lookup_hash_table_lookup_lookuptablefindv2_table_handleD
@integer_lookup_hash_table_lookup_lookuptablefindv2_default_value	B
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	D
@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleE
Astring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	
normalization_sub_y
normalization_sqrt_x
normalization_1_sub_y
normalization_1_sqrt_x
normalization_2_sub_y
normalization_2_sqrt_x
normalization_3_sub_y
normalization_3_sqrt_x
normalization_4_sub_y
normalization_4_sqrt_x
normalization_5_sub_y
normalization_5_sqrt_x
dense_294410:Q

dense_294412:
 
dense_1_294415:

dense_1_294417:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?2integer_lookup/hash_table_Lookup/LookupTableFindV2?1string_lookup/hash_table_Lookup/LookupTableFindV2?3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
2integer_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2?integer_lookup_hash_table_lookup_lookuptablefindv2_table_handlemonth@integer_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????24
2integer_lookup/hash_table_Lookup/LookupTableFindV2?
integer_lookup/IdentityIdentity;integer_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup/Identity?
integer_lookup/bincount/ShapeShape integer_lookup/Identity:output:0*
T0	*
_output_shapes
:2
integer_lookup/bincount/Shape?
integer_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
integer_lookup/bincount/Const?
integer_lookup/bincount/ProdProd&integer_lookup/bincount/Shape:output:0&integer_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2
integer_lookup/bincount/Prod?
!integer_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!integer_lookup/bincount/Greater/y?
integer_lookup/bincount/GreaterGreater%integer_lookup/bincount/Prod:output:0*integer_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2!
integer_lookup/bincount/Greater?
integer_lookup/bincount/CastCast#integer_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
integer_lookup/bincount/Cast?
integer_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
integer_lookup/bincount/Const_1?
integer_lookup/bincount/MaxMax integer_lookup/Identity:output:0(integer_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/Max?
integer_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
integer_lookup/bincount/add/y?
integer_lookup/bincount/addAddV2$integer_lookup/bincount/Max:output:0&integer_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/add?
integer_lookup/bincount/mulMul integer_lookup/bincount/Cast:y:0integer_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2
integer_lookup/bincount/mul?
!integer_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!integer_lookup/bincount/minlength?
integer_lookup/bincount/MaximumMaximum*integer_lookup/bincount/minlength:output:0integer_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2!
integer_lookup/bincount/Maximum?
!integer_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!integer_lookup/bincount/maxlength?
integer_lookup/bincount/MinimumMinimum*integer_lookup/bincount/maxlength:output:0#integer_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2!
integer_lookup/bincount/Minimum?
integer_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2!
integer_lookup/bincount/Const_2?
%integer_lookup/bincount/DenseBincountDenseBincount integer_lookup/Identity:output:0#integer_lookup/bincount/Minimum:z:0(integer_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2'
%integer_lookup/bincount/DenseBincount?
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleflat_model_type?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????23
1string_lookup/hash_table_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup/Identity?
string_lookup/bincount/ShapeShapestring_lookup/Identity:output:0*
T0	*
_output_shapes
:2
string_lookup/bincount/Shape?
string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
string_lookup/bincount/Const?
string_lookup/bincount/ProdProd%string_lookup/bincount/Shape:output:0%string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: 2
string_lookup/bincount/Prod?
 string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 string_lookup/bincount/Greater/y?
string_lookup/bincount/GreaterGreater$string_lookup/bincount/Prod:output:0)string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2 
string_lookup/bincount/Greater?
string_lookup/bincount/CastCast"string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
string_lookup/bincount/Cast?
string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
string_lookup/bincount/Const_1?
string_lookup/bincount/MaxMaxstring_lookup/Identity:output:0'string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/Max~
string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
string_lookup/bincount/add/y?
string_lookup/bincount/addAddV2#string_lookup/bincount/Max:output:0%string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/add?
string_lookup/bincount/mulMulstring_lookup/bincount/Cast:y:0string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: 2
string_lookup/bincount/mul?
 string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2"
 string_lookup/bincount/minlength?
string_lookup/bincount/MaximumMaximum)string_lookup/bincount/minlength:output:0string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: 2 
string_lookup/bincount/Maximum?
 string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2"
 string_lookup/bincount/maxlength?
string_lookup/bincount/MinimumMinimum)string_lookup/bincount/maxlength:output:0"string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2 
string_lookup/bincount/Minimum?
string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2 
string_lookup/bincount/Const_2?
$string_lookup/bincount/DenseBincountDenseBincountstring_lookup/Identity:output:0"string_lookup/bincount/Minimum:z:0'string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2&
$string_lookup/bincount/DenseBincount?
3string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2@string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handlestorey_rangeAstring_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3string_lookup_1/hash_table_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity<string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_1/Identity?
string_lookup_1/bincount/ShapeShape!string_lookup_1/Identity:output:0*
T0	*
_output_shapes
:2 
string_lookup_1/bincount/Shape?
string_lookup_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
string_lookup_1/bincount/Const?
string_lookup_1/bincount/ProdProd'string_lookup_1/bincount/Shape:output:0'string_lookup_1/bincount/Const:output:0*
T0*
_output_shapes
: 2
string_lookup_1/bincount/Prod?
"string_lookup_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"string_lookup_1/bincount/Greater/y?
 string_lookup_1/bincount/GreaterGreater&string_lookup_1/bincount/Prod:output:0+string_lookup_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2"
 string_lookup_1/bincount/Greater?
string_lookup_1/bincount/CastCast$string_lookup_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
string_lookup_1/bincount/Cast?
 string_lookup_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 string_lookup_1/bincount/Const_1?
string_lookup_1/bincount/MaxMax!string_lookup_1/Identity:output:0)string_lookup_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/Max?
string_lookup_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2 
string_lookup_1/bincount/add/y?
string_lookup_1/bincount/addAddV2%string_lookup_1/bincount/Max:output:0'string_lookup_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/add?
string_lookup_1/bincount/mulMul!string_lookup_1/bincount/Cast:y:0 string_lookup_1/bincount/add:z:0*
T0	*
_output_shapes
: 2
string_lookup_1/bincount/mul?
"string_lookup_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"string_lookup_1/bincount/minlength?
 string_lookup_1/bincount/MaximumMaximum+string_lookup_1/bincount/minlength:output:0 string_lookup_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2"
 string_lookup_1/bincount/Maximum?
"string_lookup_1/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"string_lookup_1/bincount/maxlength?
 string_lookup_1/bincount/MinimumMinimum+string_lookup_1/bincount/maxlength:output:0$string_lookup_1/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2"
 string_lookup_1/bincount/Minimum?
 string_lookup_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2"
 string_lookup_1/bincount/Const_2?
&string_lookup_1/bincount/DenseBincountDenseBincount!string_lookup_1/Identity:output:0$string_lookup_1/bincount/Minimum:z:0)string_lookup_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2(
&string_lookup_1/bincount/DenseBincount?
normalization/subSubfloor_area_sqmnormalization_sub_y*
T0*'
_output_shapes
:?????????2
normalization/subo
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
normalization_1/subSubdist_to_nearest_stnnormalization_1_sub_y*
T0*'
_output_shapes
:?????????2
normalization_1/subu
normalization_1/SqrtSqrtnormalization_1_sqrt_x*
T0*
_output_shapes

:2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_1/Maximum/y?
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_1/Maximum?
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_1/truediv?
normalization_2/subSubdist_to_dhobynormalization_2_sub_y*
T0*'
_output_shapes
:?????????2
normalization_2/subu
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes

:2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_2/Maximum/y?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_2/Maximum?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_2/truediv?
normalization_3/subSubdegree_centralitynormalization_3_sub_y*
T0*'
_output_shapes
:?????????2
normalization_3/subu
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_3/truediv?
normalization_4/subSubeigenvector_centralitynormalization_4_sub_y*
T0*'
_output_shapes
:?????????2
normalization_4/subu
normalization_4/SqrtSqrtnormalization_4_sqrt_x*
T0*
_output_shapes

:2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_4/Maximum/y?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_4/Maximum?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_4/truediv?
normalization_5/subSubremaining_lease_yearsnormalization_5_sub_y*
T0*'
_output_shapes
:?????????2
normalization_5/subu
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes

:2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_5/Maximum/y?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_5/Maximum?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_5/truediv?
concatenate/PartitionedCallPartitionedCall.integer_lookup/bincount/DenseBincount:output:0-string_lookup/bincount/DenseBincount:output:0/string_lookup_1/bincount/DenseBincount:output:0normalization/truediv:z:0normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Q* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2938712
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_294410dense_294412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2938842
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_294415dense_1_294417*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2939002!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall3^integer_lookup/hash_table_Lookup/LookupTableFindV22^string_lookup/hash_table_Lookup/LookupTableFindV24^string_lookup_1/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2h
2integer_lookup/hash_table_Lookup/LookupTableFindV22integer_lookup/hash_table_Lookup/LookupTableFindV22f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV22j
3string_lookup_1/hash_table_Lookup/LookupTableFindV23string_lookup_1/hash_table_Lookup/LookupTableFindV2:N J
'
_output_shapes
:?????????

_user_specified_namemonth:XT
'
_output_shapes
:?????????
)
_user_specified_nameflat_model_type:UQ
'
_output_shapes
:?????????
&
_user_specified_namestorey_range:WS
'
_output_shapes
:?????????
(
_user_specified_namefloor_area_sqm:\X
'
_output_shapes
:?????????
-
_user_specified_namedist_to_nearest_stn:VR
'
_output_shapes
:?????????
'
_user_specified_namedist_to_dhoby:ZV
'
_output_shapes
:?????????
+
_user_specified_namedegree_centrality:_[
'
_output_shapes
:?????????
0
_user_specified_nameeigenvector_centrality:^Z
'
_output_shapes
:?????????
/
_user_specified_nameremaining_lease_years:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:"?L
saver_filename:0StatefulPartitionedCall_4:0StatefulPartitionedCall_58"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
degree_centrality:
#serving_default_degree_centrality:0?????????
G
dist_to_dhoby6
serving_default_dist_to_dhoby:0?????????
S
dist_to_nearest_stn<
%serving_default_dist_to_nearest_stn:0?????????
Y
eigenvector_centrality?
(serving_default_eigenvector_centrality:0?????????
K
flat_model_type8
!serving_default_flat_model_type:0?????????
I
floor_area_sqm7
 serving_default_floor_area_sqm:0?????????
7
month.
serving_default_month:0	?????????
W
remaining_lease_years>
'serving_default_remaining_lease_years:0?????????
E
storey_range5
serving_default_storey_range:0?????????=
dense_12
StatefulPartitionedCall_3:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-0

layer-9
layer_with_weights-1
layer-10
layer_with_weights-2
layer-11
layer_with_weights-3
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
b
lookup_table
token_counts
	keras_api
?_adapt_function"
_tf_keras_layer
b
lookup_table
 token_counts
!	keras_api
?_adapt_function"
_tf_keras_layer
b
"lookup_table
#token_counts
$	keras_api
?_adapt_function"
_tf_keras_layer
?
%
_keep_axis
&_reduce_axis
'_reduce_axis_mask
(_broadcast_shape
)mean
)
adapt_mean
*variance
*adapt_variance
	+count
,	keras_api
?_adapt_function"
_tf_keras_layer
?
-
_keep_axis
._reduce_axis
/_reduce_axis_mask
0_broadcast_shape
1mean
1
adapt_mean
2variance
2adapt_variance
	3count
4	keras_api
?_adapt_function"
_tf_keras_layer
?
5
_keep_axis
6_reduce_axis
7_reduce_axis_mask
8_broadcast_shape
9mean
9
adapt_mean
:variance
:adapt_variance
	;count
<	keras_api
?_adapt_function"
_tf_keras_layer
?
=
_keep_axis
>_reduce_axis
?_reduce_axis_mask
@_broadcast_shape
Amean
A
adapt_mean
Bvariance
Badapt_variance
	Ccount
D	keras_api
?_adapt_function"
_tf_keras_layer
?
E
_keep_axis
F_reduce_axis
G_reduce_axis_mask
H_broadcast_shape
Imean
I
adapt_mean
Jvariance
Jadapt_variance
	Kcount
L	keras_api
?_adapt_function"
_tf_keras_layer
?
M
_keep_axis
N_reduce_axis
O_reduce_axis_mask
P_broadcast_shape
Qmean
Q
adapt_mean
Rvariance
Radapt_variance
	Scount
T	keras_api
?_adapt_function"
_tf_keras_layer
?
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Ykernel
Zbias
[regularization_losses
\	variables
]trainable_variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

_kernel
`bias
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
eiter

fbeta_1

gbeta_2
	hdecay
ilearning_rateYm?Zm?_m?`m?Yv?Zv?_v?`v?"
	optimizer
 "
trackable_list_wrapper
?
)3
*4
+5
16
27
38
99
:10
;11
A12
B13
C14
I15
J16
K17
Q18
R19
S20
Y21
Z22
_23
`24"
trackable_list_wrapper
<
Y0
Z1
_2
`3"
trackable_list_wrapper
?
jlayer_metrics

klayers
regularization_losses
lnon_trainable_variables
mmetrics
	variables
nlayer_regularization_losses
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
U
o_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
U
p_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
U
q_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
rlayer_metrics

slayers
Uregularization_losses
tnon_trainable_variables
umetrics
V	variables
vlayer_regularization_losses
Wtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:Q
2dense/kernel
:
2
dense/bias
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
?
wlayer_metrics

xlayers
[regularization_losses
ynon_trainable_variables
zmetrics
\	variables
{layer_regularization_losses
]trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
?
|layer_metrics

}layers
aregularization_losses
~non_trainable_variables
metrics
b	variables
 ?layer_regularization_losses
ctrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
?
)3
*4
+5
16
27
38
99
:10
;11
A12
B13
C14
I15
J16
K17
Q18
R19
S20"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
#:!Q
2Adam/dense/kernel/m
:
2Adam/dense/bias/m
%:#
2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
#:!Q
2Adam/dense/kernel/v
:
2Adam/dense/bias/v
%:#
2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
?2?
A__inference_model_layer_call_and_return_conditional_losses_294752
A__inference_model_layer_call_and_return_conditional_losses_294887
A__inference_model_layer_call_and_return_conditional_losses_294421
A__inference_model_layer_call_and_return_conditional_losses_294552?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_293726monthflat_model_typestorey_rangefloor_area_sqmdist_to_nearest_stndist_to_dhobydegree_centralityeigenvector_centralityremaining_lease_years	"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_model_layer_call_fn_293954
&__inference_model_layer_call_fn_294944
&__inference_model_layer_call_fn_295001
&__inference_model_layer_call_fn_294290?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_adapt_step_7663?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_7677?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_7691?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_7738?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_7785?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_7832?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_7879?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_7926?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_7973?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_concatenate_layer_call_and_return_conditional_losses_295015?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_layer_call_fn_295028?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_295039?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_295048?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_295058?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_1_layer_call_fn_295067?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_294617degree_centralitydist_to_dhobydist_to_nearest_stneigenvector_centralityflat_model_typefloor_area_sqmmonthremaining_lease_yearsstorey_range"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_295072?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_295080?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_295085?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_295090?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_295095?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_295100?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_295185checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_295193restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?	
	?	
?2?
__inference__creator_295105?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_295113?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_295118?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_295123?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_295128?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_295133?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_295212checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_295220restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
?2?
__inference__creator_295138?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_295146?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_295151?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_295156?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_295161?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_295166?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_295239checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_295247restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8
J	
Const_9
J

Const_10
J

Const_11
J

Const_12
J

Const_13
J

Const_14
J

Const_15
J

Const_16
J

Const_17
J

Const_18
J

Const_19
J

Const_20
J

Const_21
J

Const_22
J

Const_237
__inference__creator_295072?

? 
? "? 7
__inference__creator_295090?

? 
? "? 7
__inference__creator_295105?

? 
? "? 7
__inference__creator_295123?

? 
? "? 7
__inference__creator_295138?

? 
? "? 7
__inference__creator_295156?

? 
? "? 9
__inference__destroyer_295085?

? 
? "? 9
__inference__destroyer_295100?

? 
? "? 9
__inference__destroyer_295118?

? 
? "? 9
__inference__destroyer_295133?

? 
? "? 9
__inference__destroyer_295151?

? 
? "? 9
__inference__destroyer_295166?

? 
? "? B
__inference__initializer_295080???

? 
? "? ;
__inference__initializer_295095?

? 
? "? B
__inference__initializer_295113???

? 
? "? ;
__inference__initializer_295128?

? 
? "? B
__inference__initializer_295146"???

? 
? "? ;
__inference__initializer_295161?

? 
? "? ?
!__inference__wrapped_model_293726?%??"?????????????YZ_`???
???
???
?
month?????????	
)?&
flat_model_type?????????
&?#
storey_range?????????
(?%
floor_area_sqm?????????
-?*
dist_to_nearest_stn?????????
'?$
dist_to_dhoby?????????
+?(
degree_centrality?????????
0?-
eigenvector_centrality?????????
/?,
remaining_lease_years?????????
? "1?.
,
dense_1!?
dense_1?????????k
__inference_adapt_step_7663L?A?>
7?4
2?/?
??????????	IteratorSpec
? "
 k
__inference_adapt_step_7677L ?A?>
7?4
2?/?
??????????IteratorSpec
? "
 k
__inference_adapt_step_7691L#?A?>
7?4
2?/?
??????????IteratorSpec
? "
 k
__inference_adapt_step_7738L+)*A?>
7?4
2?/?
??????????IteratorSpec
? "
 k
__inference_adapt_step_7785L312A?>
7?4
2?/?
??????????IteratorSpec
? "
 k
__inference_adapt_step_7832L;9:A?>
7?4
2?/?
??????????IteratorSpec
? "
 k
__inference_adapt_step_7879LCABA?>
7?4
2?/?
??????????IteratorSpec
? "
 k
__inference_adapt_step_7926LKIJA?>
7?4
2?/?
??????????IteratorSpec
? "
 k
__inference_adapt_step_7973LSQRA?>
7?4
2?/?
??????????IteratorSpec
? "
 ?
G__inference_concatenate_layer_call_and_return_conditional_losses_295015????
???
???
"?
inputs/0?????????
"?
inputs/1?????????,
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
? "%?"
?
0?????????Q
? ?
,__inference_concatenate_layer_call_fn_295028????
???
???
"?
inputs/0?????????
"?
inputs/1?????????,
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
? "??????????Q?
C__inference_dense_1_layer_call_and_return_conditional_losses_295058\_`/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? {
(__inference_dense_1_layer_call_fn_295067O_`/?,
%?"
 ?
inputs?????????

? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_295039\YZ/?,
%?"
 ?
inputs?????????Q
? "%?"
?
0?????????

? y
&__inference_dense_layer_call_fn_295048OYZ/?,
%?"
 ?
inputs?????????Q
? "??????????
?
A__inference_model_layer_call_and_return_conditional_losses_294421?%??"?????????????YZ_`???
???
???
?
month?????????	
)?&
flat_model_type?????????
&?#
storey_range?????????
(?%
floor_area_sqm?????????
-?*
dist_to_nearest_stn?????????
'?$
dist_to_dhoby?????????
+?(
degree_centrality?????????
0?-
eigenvector_centrality?????????
/?,
remaining_lease_years?????????
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_294552?%??"?????????????YZ_`???
???
???
?
month?????????	
)?&
flat_model_type?????????
&?#
storey_range?????????
(?%
floor_area_sqm?????????
-?*
dist_to_nearest_stn?????????
'?$
dist_to_dhoby?????????
+?(
degree_centrality?????????
0?-
eigenvector_centrality?????????
/?,
remaining_lease_years?????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_294752?%??"?????????????YZ_`???
???
???
"?
inputs/0?????????	
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_294887?%??"?????????????YZ_`???
???
???
"?
inputs/0?????????	
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
p

 
? "%?"
?
0?????????
? ?
&__inference_model_layer_call_fn_293954?%??"?????????????YZ_`???
???
???
?
month?????????	
)?&
flat_model_type?????????
&?#
storey_range?????????
(?%
floor_area_sqm?????????
-?*
dist_to_nearest_stn?????????
'?$
dist_to_dhoby?????????
+?(
degree_centrality?????????
0?-
eigenvector_centrality?????????
/?,
remaining_lease_years?????????
p 

 
? "???????????
&__inference_model_layer_call_fn_294290?%??"?????????????YZ_`???
???
???
?
month?????????	
)?&
flat_model_type?????????
&?#
storey_range?????????
(?%
floor_area_sqm?????????
-?*
dist_to_nearest_stn?????????
'?$
dist_to_dhoby?????????
+?(
degree_centrality?????????
0?-
eigenvector_centrality?????????
/?,
remaining_lease_years?????????
p

 
? "???????????
&__inference_model_layer_call_fn_294944?%??"?????????????YZ_`???
???
???
"?
inputs/0?????????	
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
p 

 
? "???????????
&__inference_model_layer_call_fn_295001?%??"?????????????YZ_`???
???
???
"?
inputs/0?????????	
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
p

 
? "??????????z
__inference_restore_fn_295193YK?H
A?>
?
restored_tensors_0	
?
restored_tensors_1	
? "? z
__inference_restore_fn_295220Y K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? z
__inference_restore_fn_295247Y#K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_295185?&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor	
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
__inference_save_fn_295212? &?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
__inference_save_fn_295239?#&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
$__inference_signature_wrapper_294617?%??"?????????????YZ_`???
? 
???
@
degree_centrality+?(
degree_centrality?????????
8
dist_to_dhoby'?$
dist_to_dhoby?????????
D
dist_to_nearest_stn-?*
dist_to_nearest_stn?????????
J
eigenvector_centrality0?-
eigenvector_centrality?????????
<
flat_model_type)?&
flat_model_type?????????
:
floor_area_sqm(?%
floor_area_sqm?????????
(
month?
month?????????	
H
remaining_lease_years/?,
remaining_lease_years?????????
6
storey_range&?#
storey_range?????????"1?.
,
dense_1!?
dense_1?????????