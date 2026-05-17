import { NavLink, Outlet } from "react-router-dom";

export default function Layout() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <h1>Polytech Masters</h1>
          <p>Контуры датчиков · Registry</p>
        </div>
        <nav className="nav">
          <NavLink to="/" end>
            Обзор
          </NavLink>
          <NavLink to="/registries">Реестры</NavLink>
          <NavLink to="/stream">Mock-стрим</NavLink>
        </nav>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
