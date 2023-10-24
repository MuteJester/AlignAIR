let inwindow_sequence = null
let myBarChart;
let scatterChart;
let customPointDatasetIndex = null;

class DNASequence {
    constructor(data) {
        this.d_end = data.d_end;
        this.d_start = data.d_start;
        this.j_end = data.j_end;
        this.j_start = data.j_start;
        this.mutation_positions = data.mutation_positions;
        this.sequence = data.sequence;
        this.v_end = data.v_end;
        this.v_start = data.v_start;
        this.current_length = data.sequence.length
    }

    applyModifiedSequence(modifiedSequence) {
        let originalIndex = 0;
        let modifiedIndex = 0;

        while (originalIndex < this.sequence.length && modifiedIndex < modifiedSequence.length) {
            if (this.sequence[originalIndex] !== modifiedSequence[modifiedIndex]) {
                if (originalIndex + 1 < this.sequence.length && this.sequence[originalIndex + 1] === modifiedSequence[modifiedIndex]) {
                    // Deletion in the original sequence
                    this.adjustPositions(originalIndex, -1);
                    originalIndex++;
                } else if (modifiedIndex + 1 < modifiedSequence.length && modifiedSequence[modifiedIndex + 1] === this.sequence[originalIndex]) {
                    // Addition in the modified sequence
                    this.adjustPositions(originalIndex, 1);
                    modifiedIndex++;
                }
            }
            originalIndex++;
            modifiedIndex++;
        }

        // Handle any remaining additions at the end
        if (modifiedIndex < modifiedSequence.length) {
            this.adjustPositions(originalIndex, modifiedSequence.length - modifiedIndex);
        }

        this.sequence = modifiedSequence;
    }

    adjustPositions(index, change) {
        const adjust = (pos) => {
            if (pos > index) {
                pos += change;
            }
            return pos;
        };

        this.v_start = adjust(this.v_start);
        this.v_end = adjust(this.v_end);
        this.d_start = adjust(this.d_start);
        this.d_end = adjust(this.d_end);
        this.j_start = adjust(this.j_start);
        this.j_end = adjust(this.j_end);

        this.mutation_positions = this.mutation_positions.map(adjust).filter(pos => pos >= 0 && pos < this.sequence.length);
    }
}

function populateDropdown(contents) {
    const lines = contents.split('\n'); // Split by newline to get each option
    const dropdown = document.getElementById('dropdown');

    // Clear any existing options
    dropdown.innerHTML = '';

    lines.forEach(line => {
        const option = document.createElement('option');
        option.value = line;
        option.textContent = line;
        dropdown.appendChild(option);
    });
}


function setCaret(id, size) {
    let el = document.getElementById(id);
    const range = document.createRange();
    const sel = window.getSelection();

    range.setStart(el.childNodes[size], 1)
    range.collapse(true)

    sel.removeAllRanges()
    sel.addRange(range)
}

async function fetch_allele(type) {
    try {
        const response = await fetch('/get_alleles?type=' + type);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('There was a problem with the fetch operation:', error.message);
        throw error;  // If you want to propagate the error to the caller
    }
}


async function fetch_v_allele_latent() {
    try {
        const response = await fetch('/get_v_latent');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const responseData = await response.json();

        localStorage.setItem("v_allele_latent", JSON.stringify(responseData))

    } catch (error) {
        console.error('There was a problem with the fetch operation:', error.message);
        throw error;  // If you want to propagate the error to the caller
    }
}


function populateDropdown(dropdownId, data) {
    const dropdown = document.getElementById(dropdownId);
    dropdown.innerHTML = '';  // Clear existing options

    for (const item of data) {
        const option = document.createElement('option');
        option.value = item;
        option.textContent = item;
        dropdown.appendChild(option);
    }
}

async function loadDropdownData() {
    try {
        const vData = await fetch_allele('V');
        populateDropdown('v_allele_select', vData.V);

        const dData = await fetch_allele('D');
        populateDropdown('d_allele_select', dData.D);

        const jData = await fetch_allele('J');
        populateDropdown('j_allele_select', jData.J);
    } catch (error) {
        console.error('Error fetching allele data:', error);
    }
}


async function fetch_sequence() {
    let current_char;
    try {
        const vValue = document.getElementById('v_allele_select').value;
        const dValue = document.getElementById('d_allele_select').value;
        const jValue = document.getElementById('j_allele_select').value;

        const response = await fetch('/generate_sequence', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({"V": vValue, "D": dValue, "J": jValue})
        });

        const responseData = await response.json();


        // fill in ground truth
        const table = document.getElementById("data-table");
        table.rows[1].cells[1].innerText = vValue;
        table.rows[1].cells[2].innerText = dValue;
        table.rows[1].cells[3].innerText = jValue;
        table.rows[1].cells[4].innerText = responseData['v_start'];
        table.rows[1].cells[5].innerText = responseData['v_end'];
        table.rows[1].cells[6].innerText = responseData['d_start'];
        table.rows[1].cells[7].innerText = responseData['d_end'];
        table.rows[1].cells[8].innerText = responseData['j_start'];
        table.rows[1].cells[9].innerText = responseData['j_end'];

        inwindow_sequence = new DNASequence(responseData);


        // color the sequence properly based on VDJ and mutations
        let tempContainer = document.createElement('div');
        let tempSeq = responseData['sequence'];
        console.clear();
        for (let pos = 0; pos < tempSeq.length; pos++) {
            let span = document.createElement('span');
            current_char = tempSeq[pos]
            //V
            if (responseData['mutation_positions'].includes(pos)) {
                span.style.color = 'red';
                //span.style.fontFamily = 'New Times Roman'
                span.style.border = '1px dashed red';
                span.innerText = current_char;
                span.title = pos;
                tempContainer.appendChild(span);
            } else if (pos >= responseData['v_start'] && pos <= responseData['v_end']) {
                span.style.color = '#07b126';
                span.innerText = current_char;
                span.title = pos;
                tempContainer.appendChild(span);

            }
            //Jun1
            else if (pos > responseData['v_end'] && pos < responseData['d_start']) {
                span.style.color = 'black';
                span.style.fontWeight = 'bold';
                span.innerText = current_char;
                span.title = pos;
                tempContainer.appendChild(span);
            }
            //D
            else if (pos >= responseData['d_start'] && pos <= responseData['d_end']) {
                span.style.color = '#5733FF';
                span.innerText = current_char;
                span.title = pos;
                tempContainer.appendChild(span);
            }
            //Jun2
            else if (pos > responseData['d_end'] && pos < responseData['j_start']) {
                span.style.color = 'black';
                span.style.fontWeight = 'bold';
                span.innerText = current_char;
                span.title = pos;
                tempContainer.appendChild(span);
            }
            //J
            else if (pos >= responseData['j_start'] && pos <= responseData['j_end']) {
                span.style.color = '#f58231';
                span.innerText = current_char;
                span.title = pos;
                tempContainer.appendChild(span);
            }
            document.getElementById('sequence').innerHTML = tempContainer.innerHTML;

        }
    } catch (error) {
        console.error('Error fetching sequence:', error);
    }
}


function update_sequence_info() {
    const div = document.getElementById('sequence');
    const selection = window.getSelection();
    let startPosition = null
    if (selection.rangeCount >= 0) {
        const range = selection.getRangeAt(0);

        // Create a new range that starts at the beginning of the div and ends at the start of the selection
        const preSelectionRange = document.createRange();
        preSelectionRange.setStart(div, 0);
        preSelectionRange.setEnd(range.startContainer, range.startOffset);

        // The length of the text in the preSelectionRange is the start position of the pasted text relative to all text in the div
        startPosition = preSelectionRange.toString().length;

    }
    let lengthDiff = inwindow_sequence.current_length - div.innerText.length - 1
    startPosition -= 1;
    console.log(lengthDiff, div.innerText.length, inwindow_sequence.current_length, startPosition)

    const cols = ['v_start', 'v_end', 'd_start', 'd_end', 'j_start', 'j_end'];
    if (startPosition < inwindow_sequence['v_start']) {
        if (lengthDiff < 0) {//added
            for (let col = 0; col < cols.length; col++) {
                inwindow_sequence[cols[col]] += 1;
            }
            inwindow_sequence.current_length += 1
        } else {
            for (let col = 0; col < cols.length; col++) {
                inwindow_sequence[cols[col]] -= 1;
            }
            inwindow_sequence.current_length -= 1
        }
    } else if (startPosition > inwindow_sequence['v_start'] && startPosition < inwindow_sequence['v_end']) {
        if (lengthDiff < 0) {//added
            for (let col = 1; col < cols.length; col++) {
                inwindow_sequence[cols[col]] += 1;
            }
            inwindow_sequence.current_length += 1
        } else {
            for (let col = 1; col < cols.length; col++) {
                inwindow_sequence[cols[col]] -= 1;
            }
            inwindow_sequence.current_length -= 1
        }
    } else if (startPosition > inwindow_sequence['v_end'] && startPosition < inwindow_sequence['d_start']) {
        if (lengthDiff < 0) {//added
            for (let col = 2; col < cols.length; col++) {
                inwindow_sequence[cols[col]] += 1;
            }
            inwindow_sequence.current_length += 1
        } else {
            for (let col = 2; col < cols.length; col++) {
                inwindow_sequence[cols[col]] -= 1;
            }
            inwindow_sequence.current_length -= 1
        }
    } else if (startPosition > inwindow_sequence['d_start'] && startPosition < inwindow_sequence['d_end']) {
        if (lengthDiff < 0) {//added
            for (let col = 3; col < cols.length; col++) {
                inwindow_sequence[cols[col]] += 1;
            }
            inwindow_sequence.current_length += 1
        } else {
            for (let col = 3; col < cols.length; col++) {
                inwindow_sequence[cols[col]] -= 1;
            }
            inwindow_sequence.current_length -= 1
        }
    } else if (startPosition > inwindow_sequence['d_end'] && startPosition < inwindow_sequence['j_start']) {
        if (lengthDiff < 0) {//added
            for (let col = 4; col < cols.length; col++) {
                inwindow_sequence[cols[col]] += 1;
            }
            inwindow_sequence.current_length += 1
        } else {
            for (let col = 4; col < cols.length; col++) {
                inwindow_sequence[cols[col]] -= 1;
            }
            inwindow_sequence.current_length -= 1
        }
    } else if (startPosition > inwindow_sequence['j_start'] && startPosition < inwindow_sequence['j_end']) {
        if (lengthDiff < 0) {//added
            for (let col = 5; col < cols.length; col++) {
                inwindow_sequence[cols[col]] += 1;
            }
            inwindow_sequence.current_length += 1
        } else {
            for (let col = 5; col < cols.length; col++) {
                inwindow_sequence[cols[col]] -= 1;
            }
            inwindow_sequence.current_length -= 1
        }
    }


    //inwindow_sequence.applyModifiedSequence(document.getElementById('sequence').innerText);
    const table = document.getElementById("data-table");
    table.rows[1].cells[4].innerText = inwindow_sequence['v_start'];
    table.rows[1].cells[5].innerText = inwindow_sequence['v_end'];
    table.rows[1].cells[6].innerText = inwindow_sequence['d_start'];
    table.rows[1].cells[7].innerText = inwindow_sequence['d_end'];
    table.rows[1].cells[8].innerText = inwindow_sequence['j_start'];
    table.rows[1].cells[9].innerText = inwindow_sequence['j_end'];

}

function renderBarChart(data) {
    if (!data) {
        console.error('No data provided for bar chart rendering.');
        return;
    }


    const ctx = document.getElementById('Top5Probas_barChart').getContext('2d');
    document.getElementById('Top5Probas_barChart').style.backgroundColor = 'white'
    // If a chart instance already exists, destroy it
    if (myBarChart) {
        myBarChart.destroy();
    }

    // Create a new chart instance
    myBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(item => item.label), // Assuming each data item has a 'label' property
            datasets: [{
                label: 'Top 5 Probabilities',
                data: data.map(item => item.value), // Assuming each data item has a 'value' property
                backgroundColor: 'rgba(75, 192, 192, 1)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                x: {
                    beginAtZero: true
                },
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    document.getElementById('data-table').style.height = document.getElementById('Top5Probas_barChart').style.height
}


async function predict_sequence() {
    try {
        const seqValue = document.getElementById('sequence').innerText;

        const GTtable = document.getElementById("data-table");

        const VStart = GTtable.rows[1].cells[4].innerText;
        const Vend = GTtable.rows[1].cells[5].innerText;

        const response = await fetch('/predict_sequence', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({"sequence": seqValue,
                'v_start':VStart,
                'v_end':Vend}
            )
        });

        const responseData = await response.json();


        // fill in ground truth
        const table = document.getElementById("data-table");
        table.rows[2].cells[1].innerText = responseData['V'];
        table.rows[2].cells[2].innerText = responseData['D'];
        table.rows[2].cells[3].innerText = responseData['J'];
        table.rows[2].cells[4].innerText = responseData['v_start'];
        table.rows[2].cells[5].innerText = responseData['v_end'];
        table.rows[2].cells[6].innerText = responseData['d_start'];
        table.rows[2].cells[7].innerText = responseData['d_end'];
        table.rows[2].cells[8].innerText = responseData['j_start'];
        table.rows[2].cells[9].innerText = responseData['j_end'];


        // color the sequence properly based on VDJ and mutations
        let tempContainer = document.createElement('div');
        let tempSeq = responseData['sequence'];

        //document.getElementById('sequence').innerHTML = tempContainer.innerHTML;
        renderBarChart(responseData['top_5_probas'])

        addOrUpdateCustomPoint(responseData['v_latent_x'], responseData['v_latent_y'], 'Your Sequence')

        let sampleSequences = responseData['allgined_seqs']

    ;
        displayMSA('msaContainer',sampleSequences)
    } catch (error) {
        console.error('Error fetching sequence:', error);
    }
}

function plotScatterFromLocalStorage() {
    const data = getLocalStorageData('v_allele_latent');
    if (!data) return;

    const uniqueLabels = [...new Set(data.label)];
    const labelColors = generateLabelColors(uniqueLabels);
    const datasets = generateDatasets(uniqueLabels, data, labelColors);

    createScatterPlot(datasets);
}

function getLocalStorageData(key) {
    const dataString = localStorage.getItem(key);
    if (!dataString) {
        console.error(`No data found in localStorage under key "${key}"`);
        return null;
    }

    const data = JSON.parse(dataString);
    if (!data.x || !data.y || !data.label) {
        console.error('Data format in localStorage is incorrect');
        return null;
    }

    return data;
}

function generateLabelColors(uniqueLabels) {
    const colorPalette = [
        'rgba(0,0,0,0.7)', 'rgba(0,130,200,0.7)', 'rgba(245,130,48,0.7)',
        'rgba(255,255,25,0.7)', 'rgba(220,190,255,0.7)', 'rgba(128,0,0,0.7)',
        'rgba(0,0,128,0.7)', 'rgba(128,128,128,0.7)'
    ];

    const labelColors = {};
    uniqueLabels.forEach((label, index) => {
        labelColors[label] = colorPalette[index % colorPalette.length];
    });

    return labelColors;
}

function generateDatasets(uniqueLabels, data, labelColors) {
    return uniqueLabels.map((label) => {
        const dataPoints = data.x.map((xValue, idx) => {
            return data.label[idx] === label ? {
                x: xValue,
                y: data.y[idx],
                label: data.allele[idx]
            } : null;
        }).filter(point => point);

        return {
            label: label.replace('F', 'IGHVF'),
            data: dataPoints,
            backgroundColor: labelColors[label],
            borderColor: labelColors[label].replace('0.7', '1'),
            borderWidth: 1,
            pointRadius: 4,
            pointHoverRadius: 7,
            pointHoverBackgroundColor: 'rgba(75, 192, 192, 0.2)',
            pointHoverBorderColor: 'rgba(75, 192, 192, 1)'
        };
    });
}

function createScatterPlot(datasets) {
    const ctx = document.getElementById('V-Latent').getContext('2d');
    scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {datasets},
        options: {
            scales: {x: {beginAtZero: true}, y: {beginAtZero: true}},
            legend: {display: true, position: 'top'},
            plugins: {
                tooltip: {
                callbacks: {
                    label: (tooltipItem) => {
                        return tooltipItem.dataset.data[tooltipItem.dataIndex].label;
                    }
                }
            },

                zoom: {
                    pan: {enabled: true, mode: 'xy'},
                    zoom: {wheel: {enabled: true}, mode: 'xy'}
                }
            }
        }
    });
}

function addOrUpdateCustomPoint(x, y, hoverLabel) {

    console.log(x,y,hoverLabel)
    // Check if the scatterChart is defined
    if (!scatterChart) {
        console.error("Scatter chart is not initialized.");
        return;
    }

    // Remove the previous custom point if it exists
    if (customPointDatasetIndex !== null) {
        scatterChart.data.datasets.splice(customPointDatasetIndex, 1);
        customPointDatasetIndex = null;
    }

    // Create the new custom point dataset
    const customPointDataset = {
        label: hoverLabel,
        data: [{ x, y, label: hoverLabel }],
        backgroundColor: 'red',
        borderColor: 'red',
        borderWidth: 2,
        pointRadius: 6,
        pointHoverRadius: 7,
        pointStyle: 'cross', // This will make the point appear as an "X"
        pointHoverBackgroundColor: 'rgba(75, 192, 192, 0.2)',
        pointHoverBorderColor: 'rgba(75, 192, 192, 1)'
    };

    // Add the new custom point to the scatter plot
    scatterChart.data.datasets.push(customPointDataset);
    customPointDatasetIndex = scatterChart.data.datasets.length - 1;

    // Update the scatter plot to reflect the changes
    scatterChart.update();
}

function displayMSA(containerId, sequences) {
    var container = document.getElementById(containerId);
    container.innerHTML = '';

    // Determine the maximum sequence length for column labels
    var maxLength = Math.max(...sequences.map(seq => seq.data.length));

    // Add column labels
    var columnLabels = document.createElement('div');
    columnLabels.className = 'msa-column-labels';

    // Placeholder for sequence ID to align columns correctly
    var idPlaceholder = document.createElement('div');
    idPlaceholder.className = 'msa-id-placeholder';
    columnLabels.appendChild(idPlaceholder);

    for (let i = 1; i <= maxLength; i++) {
        var label = document.createElement('div');
        label.className = 'msa-column-label';
        label.textContent = i;
        columnLabels.appendChild(label);
    }
    container.appendChild(columnLabels);

    sequences.forEach(function(sequence, index) {
        var row = document.createElement('div');
        row.className = 'msa-row';
        row.draggable = true;
        row.dataset.index = index;

        // Add sequence ID as a label to the left
        var idLabel = document.createElement('label');
        idLabel.className = 'msa-id';
        idLabel.textContent = sequence.id;
        row.appendChild(idLabel);

       sequence.data.split('').forEach(function(nucleotide) {
        var cell = document.createElement('div');
        // Adjust the class assignment based on the renamed CSS class names
        if (nucleotide === '-') {
            cell.className = 'msa-cell pad_token'; // Use 'pad_token' for gaps/pads
        } else {
            cell.className = 'msa-cell ' + nucleotide + '_token'; // Append '_token' to the nucleotide name
        }
        cell.textContent = nucleotide;
        row.appendChild(cell);
    });
        container.appendChild(row);
    });

    // Drag and drop functionality for rows
    let draggedItem = null;
    container.addEventListener('dragstart', function(e) {
        draggedItem = e.target;
        setTimeout(function() {
            draggedItem.style.display = 'none';
        }, 0);
    });

    container.addEventListener('dragend', function(e) {
        setTimeout(function() {
            draggedItem.style.display = '';
            draggedItem = null;
        }, 0);
    });

    container.addEventListener('dragover', function(e) {
        e.preventDefault();
    });

    container.addEventListener('dragenter', function(e) {
        if (e.target.className === 'msa-row') {
            e.target.style.borderTop = '3px solid #666';
        }
    });

    container.addEventListener('dragleave', function(e) {
        if (e.target.className === 'msa-row') {
            e.target.style.borderTop = '';
        }
    });

    container.addEventListener('drop', function(e) {
        if (e.target.className === 'msa-row') {
            e.target.style.borderTop = '';
            container.insertBefore(draggedItem, e.target);
        }
    });
}
function toggleIdenticalColumns() {
    const checkbox = document.getElementById('removeIdenticalColumns');
    const columns = document.querySelectorAll('.msa-column-label, .msa-cell');
    const sequences = Array.from(document.querySelectorAll('.msa-row'));

    if (checkbox.checked) {
        for (let i = 0; i < sequences[0].children.length - 1; i++) { // -1 to exclude the ID label
            let identical = true;
            let value = sequences[0].children[i + 1].textContent; // +1 to exclude the ID label

            for (let j = 1; j < sequences.length; j++) {
                if (sequences[j].children[i + 1].textContent !== value) {
                    identical = false;
                    break;
                }
            }

            if (identical) {
                sequences.forEach(seq => seq.children[i + 1].style.display = 'none');
                document.querySelectorAll('.msa-column-label')[i].style.display = 'none';
            }
        }
    } else {
        columns.forEach(column => column.style.display = '');
    }
}


function myInit() {
    loadDropdownData()
    //save V allele latent
    if (localStorage.getItem("v_allele_latent") === null) {
        fetch_v_allele_latent()
    }

    plotScatterFromLocalStorage()

}

window.addEventListener("load", myInit, true);
