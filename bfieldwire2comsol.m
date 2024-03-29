function bfieldwire2comsol(model, wirefile)

linetype='pol';

selname = 'csel1';
selflag = false;

wires = {};

f = fopen(wirefile);
wire = fgets(f);
while wire>0
    coords = strsplit(wire,';');
    n = length(coords);
    stack = zeros(n,3);
    for i=1:n
        xyz = strsplit(coords{i},',');
        for j=1:3
            stack(i,j) = str2double(xyz{j});
        end
    end
    wires{end+1} = stack;
    wire = fgets(f);
end

fclose(f);

%{
Add wires to 3rd component in model.
1) Shielding
2) Coil Layer
3) Combined for simulation
%}
model.param.set('coil_i', '0.1[A]', '');


% First identify wires in model and find an adjacent index.
geonode = model.component('comp3').geom('geom3');

objectlabels = char(geonode.feature.tags);
searching=true;
wireN = 1;

while searching
    if any(ismember(objectlabels,[linetype num2str(wireN)],'rows'))
        wireN = wireN + 1;
    else
        searching = false;
    end
end

for i=1:length(wires)
    selfN = insertWire(model, wires{i}, wireN, selflag, linetype);
    selflag = false;
    if selfN ~= wireN
        warning("Winding numbers disagree")
    end
    wireN = wireN + 1;
end

if ~exist('coilindex','var') || isempty(coilindex)
  coilindex = 1;
end

coilname = ['edc' num2str(coilindex)];

mfmodel = model.component('comp3').physics('mf');

nodes = mfmodel.feature.tags;

alreadysetup = false;
for i=1:length(nodes)
    if strcmp(char(nodes(i)),coilname)
        alreadysetup=true;
    end
end

if ~alreadysetup
    disp('Energizing edge currents in model')
    mfmodel.create(coilname, 'EdgeCurrent', 1);
else
    disp('Updating edge currents in model')
end

mfmodel.feature(coilname).selection.named(['geom3_' selname '_edg']);
mfmodel.feature(coilname).set('Ie', 'coil_i');

end

function varargout = insertWire(model, contourdata, icN, newsel, linetype, whichsel)
% function insertWires(model, wiredata, tag)
% 
% Adds wires contained in wiredata to model, identified with 'tag'. If user
% doesn't specify an index for the interpolation curve 'icN', function
% searches for next available index and assigns that to wire.
% ARR 2020.06.16

if ~exist('newsel','var') || isempty(newsel)
  newsel=true;
end

if ~exist('selname','var') || isempty(selname)
  selname='csel1';
end

if ~exist('icN','var') || isempty(icN)
  icN=1;
  searching=true;
else
  searching=false;
end

if ~exist('linetype','var') || isempty(linetype)
  linetype='ic';
end

if ~exist('whichsel','var') || isempty(whichsel)
  whichsel='csel1';
end

% First identify wires in model and find an adjacent index.
geonode = model.component('comp3').geom('geom3');

if searching
    objectlabels = char(geonode.feature.tags);
end

wireN = icN;
varargout{1} = icN;

while searching
   if any(ismember(objectlabels,[linetype num2str(wireN)],'rows'))
       wireN = wireN + 1;
   else
       searching = false;
   end
end

% Now add new wire
% check to see if selection group exists yet

geonode.feature.create(['pol' num2str(wireN)], 'Polygon').set('source', 'table').set('table', contourdata).set('type', 'closed');
    
if newsel
try
    geonode.selection.create('csel1', 'CumulativeSelection');
    varargout{2}='csel1';
catch
    warning("Couldn't create cumulative selection 1.")
end
end
try
    model.component('comp3').geom('geom3').selection('csel1').label(selname);
    varargout{3} = selname;
catch
    warning(['csel naming failed: ', selname])
end

geonode.feature(['pol' num2str(wireN)]).set('contributeto', whichsel);
end

%{
Todo
 x Convert to Function
 x Add COMSOL model handling
 - Document export/import functions in Zettelkasten
%}