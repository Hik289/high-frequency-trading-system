package com.boot.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.stereotype.Repository;

import com.boot.entity.Files;

@Repository
public interface FilesRepository extends JpaRepository<Files, String>,JpaSpecificationExecutor<Files>{

}
