// Highlight navbar links when clicked

const navLinks =
  document.querySelectorAll(".nav-links a");

navLinks.forEach((link) => {

  link.addEventListener("click", () => {

    navLinks.forEach((item) =>
      item.classList.remove("active")
    );

    link.classList.add("active");

  });

});


// Library button scrolls to novels

const libraryButton =
  document.querySelector(".nav-button");

libraryButton.addEventListener("click", () => {

  document
    .querySelector("#novels")
    .scrollIntoView({
      behavior: "smooth"
    });

});