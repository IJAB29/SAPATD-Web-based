var scatter_chart_data = [
  { x: 20, y: 30 },
  { x: 40, y: 50 },
  { x: 60, y: 70 },
  // and so on...
];


var scatterChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  legend: {
    display: true
  },
  scales: {
    xAxes: [{
      scaleLabel: {
        display: true,
        labelString: 'Grade 11 GWA'
      },
      ticks: {
        min: 0,
        max: 100
      }
    }],
    yAxes: [{
      scaleLabel: {
        display: true,
        labelString: 'Grade 12 GWA'
      },
      ticks: {
        min: 0,
        max: 100
      }
    }]
  }
};

var scatterChartData = {
  datasets: [{
    label: 'Predicted Results',
    data: scatter_chart_data,
    backgroundColor: 'rgba(255, 99, 132, 0.5)'
  }]
};

function addRandomString() {
  var input = document.getElementById('file-name');
  var randomString = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  input.value += ' ' + randomString;
}

function searchTable() {
  // Declare variables
  var input, filter, table, tr, td, i, j, txtValue;
  input = document.getElementById("search-box");
  filter = input.value.toUpperCase();
  table = document.getElementById("table-body");
  tr = table.getElementsByTagName("tr");

  // Loop through all table rows, and hide those that don't match the search query
  for (i = 0; i < tr.length; i++) {
    for (j = 0; j < tr[i].cells.length; j++) {
      td = tr[i].cells[j];
      if (td) {
        txtValue = td.textContent || td.innerText;
        if (txtValue.toUpperCase().indexOf(filter) > -1) {
          tr[i].style.display = "";
          break;
        } else {
          tr[i].style.display = "none";
        }
      }
    }
  }
}
