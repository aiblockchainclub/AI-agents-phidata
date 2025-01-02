const person = {
  firstName: "John",
  lastName : "Doe",
  id       : 5566,
  fullName : function() {
    return this.firstName + " " + this.lastName;
  }
};


this.firstName = "lokesh";
this.name = "lokesh";

console.log(person.fullName()); // lokesh Doe


function myFunction() {
  return this.global.global;
}



async function hii(){

  let a ;
  const data = await  fetch('https://jsonplaceholder.typicode.com/todos/1')
  let an = await data.json();
  return an;
}

console.log(hii());
// console.log(myfunc(1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3)); // Window