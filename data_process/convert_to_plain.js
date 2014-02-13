var fs = require('fs');
var readline = require('readline');
var stream = require('stream');

// var instream = fs.createReadStream('/Users/arthurbatista/datasets/estados/NY',{ flags: 'r'});
// var outstream = new stream;
// var rl = readline.createInterface(instream, outstream);

// rl.on('line', function(line) {

// 	var str = line.replace('::',',');
// 	str = str.replace('::',',');
// 	str += '\n';

// 	fs.appendFileSync('/Users/arthurbatista/datasets/NY',str);

// });

// rl.on('close', function() {
// 	console.log('DONE!');
// });

var instream = fs.createReadStream('/home/arthur/projects/mestrado/ml/data_process/NY',{ flags: 'r'});
var outstream = new stream;
var rl = readline.createInterface(instream, outstream);

var ds_user = {}
var ds_item = {}
var matrix = [];
var i=0;
var j=0;

function size(obj) {
    var size = 0, key;
    for (key in obj) {
        if (obj.hasOwnProperty(key)) size++;
    }
    return size;
}

rl.on('line', function(line) {

	var str = line.split(',');
	var user_id = str[0];

	if(ds_user[user_id] == null)
		ds_user[user_id] = i++;

});

rl.on('close', function() {
	console.log(i);
	processItems();
});

function processItems() {
	
	instream = fs.createReadStream('/home/arthur/projects/mestrado/ml/data_process/NY',{ flags: 'r'});
	outstream = new stream;
	rl = readline.createInterface(instream, outstream);

	rl.on('line', function(line) {

		var str = line.split(',');
		var item_id = str[1];

		if(ds_item[item_id] == null)
			ds_item[item_id] = j++;

	});

	rl.on('close', function() {
		console.log(j);
		processMatrix();
	});
}

function processMatrix() {

	//Initialize matrix
	for(var y=0; y<i; y++) {
		matrix[y] = new Array();
		for(var x=0; x<j; x++) {
			matrix[y][x] = 0;
		}
	}

	console.log('aqui');

	// instream = fs.createReadStream('/home/arthur/projects/mestrado/ml/data_process/NY',{ flags: 'r'});
	// outstream = new stream;
	// rl = readline.createInterface(instream, outstream);

	// rl.on('line', function(line) {

	// 	var str = line.split(',');
		
	// 	var u_id = str[0];
	// 	var u_index = ds_user[u_id];

	// 	var i_id = str[1];
	// 	var i_index = ds_item[i_id];

	// 	matrix[u_index][i_index] = parseInt(str[2]);

	// });

	// rl.on('close', function() {
	// 	console.log(matrix);
	// });

}