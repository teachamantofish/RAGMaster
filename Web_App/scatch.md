project description: 

This is a pywebview app with an HTML UI. 
The UI has multiple tabs. 
Each tab has unique functionality. 

The index.html file is the parent page and each tab essentially calls an html fragment which is included for display. 

I am using vite as a server for development, so all functionality has to work in a browser and in pywebview. See the existing code for how this is set up--it's mostly just a matter of specifying the right file path in both locations. 

I am currently working on the source.html page/content. It includes a sortable and searchable table of csv data created with DataTables.js. Note the following: 
- UI is defined in datatables.js, including all buttons. 
- individual pages like source.html, can toggle the UI on and off by setting buttons to true/false. 
- clicking a button will invoke a function call that from datatables.js to util.js. 
- DataTables functionality is defined in datatables.js
- However, all app fuctionality (what the buttons do) is defined in util.js. 


First task: 

- Create a function that displays an alert when the Save button is clicked. 
- It should show many checkboxes were selected in the datatable displayed in source.html. 
- DataTables may have a default method for getting checked checkboxes. Research if that is true. If so, use that. If not write a new one. 

